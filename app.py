import streamlit as st
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from io import BytesIO

# ============================================================================
# CACHE DO CAMPO DE RESULTADOS
# ============================================================================
def _convolucao_1d(A, k, axis):
    """Convolução 1-D simples por eixo, sem SciPy."""
    if axis == 1:
        return np.apply_along_axis(lambda row: np.convolve(row, k, mode='same'), 1, A)
    return np.apply_along_axis(lambda col: np.convolve(col, k, mode='same'), 0, A)


def _kernel_gauss(sigma):
    sigma = float(sigma)
    if sigma <= 0:
        return np.array([1.0])
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(x*x) / (2.0 * sigma**2))
    k /= k.sum()
    return k


def suavizar_campo_visual(G, sigma=1.35, preservar_mask=True):
    """Suavização gaussiana mascarada, apenas para renderização.

    Aceita sigma escalar ou (sigma_x, sigma_z), sempre em pixels.
    A reconstrução física e os valores de Gauss permanecem inalterados.
    """
    if sigma is None:
        return G.copy()
    if np.isscalar(sigma):
        sx = sz = float(sigma)
    else:
        sx, sz = float(sigma[0]), float(sigma[1])
    if sx <= 0 and sz <= 0:
        return G.copy()

    kx = _kernel_gauss(max(sx, 1e-9))
    kz = _kernel_gauss(max(sz, 1e-9))
    finite = np.isfinite(G)
    A = np.where(finite, G, 0.0)
    W = finite.astype(float)

    B = _convolucao_1d(A, kx, axis=1)
    BW = _convolucao_1d(W, kx, axis=1)
    C = _convolucao_1d(B, kz, axis=0)
    CW = _convolucao_1d(BW, kz, axis=0)

    out = np.divide(C, CW, out=np.full_like(C, np.nan), where=CW > 1e-12)
    vals = G[finite]
    if vals.size:
        out = np.clip(out, float(np.min(vals)), float(np.max(vals)))
    if preservar_mask:
        out[~finite] = np.nan
    return out


def _dilatar_mascara(mask, n_iter=1):
    """Dilatação 4-vizinhos, sem SciPy."""
    out = mask.copy()
    for _ in range(max(0, int(n_iter))):
        up = np.zeros_like(out); up[1:] = out[:-1]
        down = np.zeros_like(out); down[:-1] = out[1:]
        left = np.zeros_like(out); left[:,1:] = out[:,:-1]
        right = np.zeros_like(out); right[:,:-1] = out[:,1:]
        out |= up | down | left | right
    return out


@st.cache_data(show_spinner=False, max_entries=64)
def reconstruir_malha_global_cache(dados_elementos, nx=720, nz=480, sigma=1.35):
    """Reconstrói o campo visual em uma grade regular de alta resolução.

    Cada ponto é avaliado localmente no Q8 do elemento que o contém.
    A continuidade visual é refinada em interfaces através de uma faixa de
    transição controlada; isso é apenas pós-processamento gráfico e não altera
    os valores originais de Gauss armazenados pelo solver.
    """
    if not dados_elementos:
        return None

    from matplotlib.path import Path as MplPath

    xs, zs = [], []
    elementos_prepared = []
    for item in dados_elementos:
        el, mat, conec, xy_flat, g_flat = item
        xy = np.asarray(xy_flat, dtype=float).reshape(8, 2)
        g = np.asarray(g_flat, dtype=float)
        poly = xy[[0, 2, 4, 6]]
        # Permite pontos de Gauss com campos indisponíveis (ex.: ******).
        # O elemento continua válido se houver pelo menos 3 valores finitos.
        if np.count_nonzero(np.isfinite(g)) < 3:
            continue
        xs.extend(poly[:, 0]); zs.extend(poly[:, 1])
        elementos_prepared.append((int(el), int(mat), xy, g, poly))

    if not elementos_prepared:
        return None

    xmin, xmax = float(min(xs)), float(max(xs))
    zmin, zmax = float(min(zs)), float(max(zs))
    if np.isclose(xmin, xmax) or np.isclose(zmin, zmax):
        return None

    nx, nz = int(nx), int(nz)
    gx = np.linspace(xmin, xmax, nx)
    gz = np.linspace(zmin, zmax, nz)
    GX, GZ = np.meshgrid(gx, gz)
    accum = np.zeros_like(GX, dtype=float)
    count = np.zeros_like(GX, dtype=float)
    mat_votes = {}

    def shape8(R, S):
        return np.array([
            -0.25*(1-R)*(1-S)*(1+R+S),
             0.50*(1-R*R)*(1-S),
            -0.25*(1+R)*(1-S)*(1-R+S),
             0.50*(1+R)*(1-S*S),
            -0.25*(1+R)*(1+S)*(1-R-S),
             0.50*(1-R*R)*(1+S),
            -0.25*(1-R)*(1+S)*(1+R-S),
             0.50*(1-R)*(1-S*S),
        ])

    gp = 1.0 / np.sqrt(3.0)

    for _, mat, xy, g, poly in elementos_prepared:
        bx0, bx1 = float(np.min(poly[:,0])), float(np.max(poly[:,0]))
        bz0, bz1 = float(np.min(poly[:,1])), float(np.max(poly[:,1]))
        i0 = max(0, int(np.searchsorted(gx, bx0, side='left')) - 1)
        i1 = min(nx - 1, int(np.searchsorted(gx, bx1, side='right')))
        j0 = max(0, int(np.searchsorted(gz, bz0, side='left')) - 1)
        j1 = min(nz - 1, int(np.searchsorted(gz, bz1, side='right')))
        if i1 < i0 or j1 < j0:
            continue

        xx = GX[j0:j1+1, i0:i1+1]
        zz = GZ[j0:j1+1, i0:i1+1]
        inside = MplPath(poly).contains_points(
            np.column_stack((xx.ravel(), zz.ravel())), radius=1e-10
        ).reshape(xx.shape)
        if not np.any(inside):
            continue

        c0, c1, c2, c3 = poly
        xc = 0.25*(c0[0]+c1[0]+c2[0]+c3[0])
        zc = 0.25*(c0[1]+c1[1]+c2[1]+c3[1])
        hx = max(0.5*(np.max(poly[:,0])-np.min(poly[:,0])), 1e-12)
        hz = max(0.5*(np.max(poly[:,1])-np.min(poly[:,1])), 1e-12)
        R = np.clip((xx-xc)/hx, -1.5, 1.5)
        S = np.clip((zz-zc)/hz, -1.5, 1.5)

        # Inversa X,Z -> r,s para a geometria Q8 real.
        for _ in range(10):
            N = shape8(R, S)
            Xcur = np.sum(N * xy[:,0,None,None], axis=0)
            Zcur = np.sum(N * xy[:,1,None,None], axis=0)
            # Derivadas analíticas
            dR = np.array([
                (-2*R-S)*(S-1)/4, R*(S-1), (-2*R+S)*(S-1)/4,
                0.5-0.5*S*S, (2*R+S)*(S+1)/4, -R*(S+1),
                (2*R-S)*(S+1)/4, 0.5*S*S-0.5])
            dS = np.array([
                (-R-2*S)*(R-1)/4, 0.5*R*R-0.5, (-R+2*S)*(R+1)/4,
                -S*(R+1), (R+1)*(R+2*S)/4, 0.5-0.5*R*R,
                (R-1)*(R-2*S)/4, S*(R-1)])
            J11 = np.sum(dR * xy[:,0,None,None], axis=0)
            J12 = np.sum(dS * xy[:,0,None,None], axis=0)
            J21 = np.sum(dR * xy[:,1,None,None], axis=0)
            J22 = np.sum(dS * xy[:,1,None,None], axis=0)
            det = J11*J22 - J12*J21
            good = np.abs(det) > 1e-12
            dX = xx - Xcur; dZ = zz - Zcur
            dRstep = np.zeros_like(R); dSstep = np.zeros_like(S)
            dRstep[good] = (dX[good]*J22[good] - dZ[good]*J12[good]) / det[good]
            dSstep[good] = (J11[good]*dZ[good] - J21[good]*dX[good]) / det[good]
            R += dRstep; S += dSstep
            if np.any(inside):
                if np.nanmax(np.abs(dRstep[inside])) < 1e-7 and np.nanmax(np.abs(dSstep[inside])) < 1e-7:
                    break

        valid = inside & np.isfinite(R) & np.isfinite(S) & (np.abs(R) <= 1.001) & (np.abs(S) <= 1.001)
        if not np.any(valid):
            continue

        # Interpolação bilinear entre os 4 pontos de Gauss.
        # Quando um Gauss estiver indisponível (******), fazemos uma
        # reconstrução local normalizada apenas com os Gauss disponíveis,
        # sem inventar um valor físico para o ponto ausente.
        lr_m = (gp - R)/(2*gp); lr_p = (gp + R)/(2*gp)
        ls_m = (gp - S)/(2*gp); ls_p = (gp + S)/(2*gp)
        W = np.stack([lr_m*ls_m, lr_p*ls_m, lr_p*ls_p, lr_m*ls_p])
        finite_g = np.isfinite(g)
        if np.all(finite_g):
            V = np.sum(g[:, None, None] * W, axis=0)
        else:
            # Reponderação apenas para renderização: os dados originais
            # permanecem NaN no ponto que veio como ******.
            Wf = W.copy()
            Wf[~finite_g, ...] = 0.0
            denom = np.sum(Wf, axis=0)
            numer = np.sum(np.where(finite_g[:, None, None], g[:, None, None], 0.0) * Wf, axis=0)
            V = np.divide(numer, denom, out=np.full_like(R, np.nan), where=np.abs(denom) > 1e-12)
        gfinite = g[finite_g]
        if gfinite.size:
            V = np.clip(V, float(np.min(gfinite)), float(np.max(gfinite)))
        V[~valid] = np.nan

        slc = (slice(j0,j1+1), slice(i0,i1+1))
        m = np.isfinite(V)
        part_a = accum[slc]; part_c = count[slc]
        part_a[m] += V[m]; part_c[m] += 1.0
        accum[slc] = part_a; count[slc] = part_c

        # Votação de material para detectar interfaces.
        votes = mat_votes.get(mat)
        if votes is None:
            votes = np.zeros_like(GX, dtype=np.int16)
            mat_votes[mat] = votes
        sub = votes[slc]
        sub[m] += 1
        votes[slc] = sub

    raw = np.divide(accum, count, out=np.full_like(accum, np.nan), where=count > 0)
    valid_global = np.isfinite(raw)
    if not np.any(valid_global):
        return None

    # Material dominante em cada célula.
    mat_grid = np.full(raw.shape, -1, dtype=np.int16)
    # Votação robusta sem truques com arrays vazios:
    best_votes = np.zeros(raw.shape, dtype=np.int16)
    for mat, votes in mat_votes.items():
        better = votes > best_votes
        mat_grid[better] = int(mat)
        best_votes[better] = votes[better]

    # Detecta interfaces entre materiais ou mudanças bruscas de suporte.
    interface = np.zeros(raw.shape, dtype=bool)
    a = mat_grid
    interface[:,1:] |= (a[:,1:] >= 0) & (a[:,:-1] >= 0) & (a[:,1:] != a[:,:-1])
    interface[1:,:] |= (a[1:,:] >= 0) & (a[:-1,:] >= 0) & (a[1:,:] != a[:-1,:])
    # Inclui bordas de elementos para deixar a transição visual contínua.
    finite_shift = np.zeros_like(valid_global)
    finite_shift[:,1:] |= valid_global[:,1:] != valid_global[:,:-1]
    finite_shift[1:,:] |= valid_global[1:,:] != valid_global[:-1,:]
    interface |= finite_shift

    # Suavização visual anisotrópica baseada no tamanho físico típico da malha.
    # Assim a suavização cresce com o tamanho do elemento, em vez de depender
    # somente da resolução da imagem.
    widths = []
    heights = []
    for _, _, _, _, ppoly in elementos_prepared:
        widths.append(float(np.max(ppoly[:, 0]) - np.min(ppoly[:, 0])))
        heights.append(float(np.max(ppoly[:, 1]) - np.min(ppoly[:, 1])))
    tamanho_tipico = float(np.median(np.minimum(widths, heights))) if widths and heights else 1.0
    dx = max((xmax - xmin) / max(nx - 1, 1), 1e-12)
    dz = max((zmax - zmin) / max(nz - 1, 1), 1e-12)
    sigma_phys = max(float(sigma), 0.10 * tamanho_tipico)
    sigma_x = sigma_phys / dx
    sigma_z = sigma_phys / dz

    smooth_global = suavizar_campo_visual(raw, sigma=(sigma_x, sigma_z), preservar_mask=True)
    band = _dilatar_mascara(
        interface,
        n_iter=max(2, int(np.ceil(0.35 * max(sigma_x, sigma_z))))
    ) & valid_global
    smooth_interface = suavizar_campo_visual(
        raw,
        sigma=(1.60 * sigma_x, 1.60 * sigma_z),
        preservar_mask=True
    )
    G = raw.copy()
    G[valid_global] = 0.94*smooth_global[valid_global] + 0.06*raw[valid_global]
    G[band] = 0.88*smooth_interface[band] + 0.12*G[band]

    # Nunca permite que a representação ultrapasse os valores físicos originais.
    raw_vals = raw[valid_global]
    G[valid_global] = np.clip(G[valid_global], float(np.min(raw_vals)), float(np.max(raw_vals)))
    G[~valid_global] = np.nan
    return tuple(GX.ravel()), tuple(GZ.ravel()), tuple(G.ravel()), int(nx), int(nz)

def grade_cache_para_mpl(tupla):
    if tupla is None:
        return None
    gx_flat, gz_flat, vals_flat, nx, nz = tupla
    GX=np.asarray(gx_flat).reshape(nz,nx)
    GZ=np.asarray(gz_flat).reshape(nz,nx)
    G=np.asarray(vals_flat).reshape(nz,nx)
    return GX,GZ,G

def interpolar_grade_regular(GX, GZ, G, xq, zq):
    """Interpolação bilinear em grade retangular regular, sem SciPy/triangulação."""
    x = GX[0, :]
    z = GZ[:, 0]
    xq = np.asarray(xq, dtype=float); zq=np.asarray(zq, dtype=float)
    out=np.full(xq.shape, np.nan, dtype=float)
    inside=(xq>=x[0])&(xq<=x[-1])&(zq>=z[0])&(zq<=z[-1])
    if not np.any(inside): return out
    ix=np.clip(np.searchsorted(x,xq[inside])-1,0,len(x)-2)
    iz=np.clip(np.searchsorted(z,zq[inside])-1,0,len(z)-2)
    x1,x2=x[ix],x[ix+1]; z1,z2=z[iz],z[iz+1]
    tx=np.divide(xq[inside]-x1,x2-x1,out=np.zeros_like(x1),where=(x2!=x1))
    tz=np.divide(zq[inside]-z1,z2-z1,out=np.zeros_like(z1),where=(z2!=z1))
    q11=G[iz,ix]; q21=G[iz,ix+1]; q12=G[iz+1,ix]; q22=G[iz+1,ix+1]
    vals=np.stack([q11,q21,q12,q22])
    ok=np.all(np.isfinite(vals),axis=0)
    v=np.full_like(tx,np.nan)
    v[ok]=(1-tx[ok])*(1-tz[ok])*q11[ok]+tx[ok]*(1-tz[ok])*q21[ok]+(1-tx[ok])*tz[ok]*q12[ok]+tx[ok]*tz[ok]*q22[ok]
    out[inside]=v
    return out


class LeitorPROGEO:
    def __init__(self, caminho_arquivo):
        self.caminho = caminho_arquivo
        self.nos = {}
        self.elementos = {}
        self.materiais = {}
        self.historico_nos = {}
        self.historico_elem = {}
        self.passo_info = {}
        self.total_passos = 0
        self.max_desloc_global = 0.0
        self.largura_barragem = 0.0
        self.regex_float = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?'
        self._parse_arquivo()
        self._calcular_escalas_base()

    def _extrair_campos_resultado(self, linha, marcador, quantidade):
        """Extrai campos numéricos preservando ****** como NaN."""
        if marcador not in linha:
            return None
        trecho = linha.split(marcador, 1)[1]
        tokens = re.findall(r'\*{6}|' + self.regex_float, trecho)
        if len(tokens) < quantidade:
            return None
        tokens = tokens[:quantidade]
        out = []
        for token in tokens:
            if token == '******':
                out.append(np.nan)
            else:
                try:
                    out.append(float(token))
                except ValueError:
                    out.append(np.nan)
        return out

    def _parse_arquivo(self):
        """Lê o .PRI assumindo o padrão fixo do PROGEO:
        4 pontos de Gauss seguidos de 1 linha média por elemento.
        """
        estagio_atual = 0
        passo_global = 0
        lendo_coord = lendo_elem = lendo_nos_result = False
        el_atual = None
        gauss_temp_S, gauss_temp_E = {}, {}

        with open(self.caminho, 'r', encoding='latin-1', errors='replace') as f:
            for linha in f:
                linha_strip = linha.strip()

                if "NODE     X CO-ORD     Z CO-ORD" in linha:
                    lendo_coord = True
                    continue
                if lendo_coord:
                    if linha_strip == "" or "BOUNDARY" in linha:
                        lendo_coord = False
                    else:
                        partes = linha.split()
                        if len(partes) >= 3 and partes[0].isdigit():
                            try:
                                self.nos[int(partes[0])] = {
                                    'X': float(partes[1]),
                                    'Z': float(partes[2])
                                }
                                self.historico_nos[int(partes[0])] = {}
                            except ValueError:
                                pass

                if "ELEMENT           C O N N E C T I O N S" in linha:
                    lendo_elem = True
                    continue
                if lendo_elem:
                    if "NO. OF ELEMENTS" in linha or linha_strip == "":
                        lendo_elem = False
                    else:
                        partes = linha.split()
                        if len(partes) >= 11 and partes[0].isdigit():
                            try:
                                id_el = int(partes[0])
                                self.elementos[id_el] = [int(p) for p in partes[1:9]]
                                self.materiais[id_el] = int(partes[10])
                                self.historico_elem.setdefault(id_el, {})
                            except ValueError:
                                pass

                match_inc = re.search(r'START OF INCREMENT NO\.\s+(\d+)', linha)
                if match_inc:
                    incremento_local = int(match_inc.group(1))
                    if incremento_local == 1:
                        estagio_atual += 1
                    passo_global += 1
                    self.total_passos = passo_global
                    self.passo_info[passo_global] = {
                        'Estagio': estagio_atual,
                        'Inc': incremento_local
                    }
                    gauss_temp_S = {el: [] for el in self.elementos.keys()}
                    gauss_temp_E = {el: [] for el in self.elementos.keys()}
                    el_atual = None

                match_el = re.search(r'EL\.NO\.\s+(\d+)', linha)
                if match_el:
                    el_atual = int(match_el.group(1))
                    gauss_temp_S.setdefault(el_atual, [])
                    gauss_temp_E.setdefault(el_atual, [])

                if el_atual is not None and " S= " in linha:
                    vals_s = self._extrair_campos_resultado(linha, "S=", 11)
                    if vals_s is not None and len(gauss_temp_S[el_atual]) < 5:
                        gauss_temp_S[el_atual].append(vals_s)

                if el_atual is not None and " E= " in linha:
                    vals_e = self._extrair_campos_resultado(linha, "E=", 8)
                    if vals_e is not None and len(gauss_temp_E[el_atual]) < 5:
                        gauss_temp_E[el_atual].append(vals_e)

                if "TOTAL NODAL VALUES" in linha:
                    lendo_nos_result = True
                    # Padrão fixo: S1..S4 = Gauss; S5 = média.
                    for el in self.elementos.keys():
                        if len(gauss_temp_S.get(el, [])) < 5 or len(gauss_temp_E.get(el, [])) < 5:
                            continue

                        g_S = np.asarray(gauss_temp_S[el][:4], dtype=float)
                        g_E = np.asarray(gauss_temp_E[el][:4], dtype=float)
                        m_S = np.asarray(gauss_temp_S[el][4], dtype=float)
                        m_E = np.asarray(gauss_temp_E[el][4], dtype=float)

                        pwp = m_S[9]
                        ev = m_E[0] + m_E[1] + m_E[2]
                        gauss_vals = {
                            'SXX': g_S[:, 0], 'SYY': g_S[:, 1], 'SZZ': g_S[:, 2], 'SXZ': g_S[:, 3],
                            'S1': g_S[:, 4], 'S3': g_S[:, 5], 'PWP': g_S[:, 9], 'RM': g_S[:, 10],
                            'EXX': g_E[:, 0], 'EYY': g_E[:, 1], 'EZZ': g_E[:, 2], 'EXZ': g_E[:, 3],
                            'E1': g_E[:, 4], 'E3': g_E[:, 5],
                            'EV': g_E[:, 0] + g_E[:, 1] + g_E[:, 2],
                            'S_DEV': g_S[:, 4] - g_S[:, 5],
                            'E_DEV': g_E[:, 4] - g_E[:, 5]
                        }

                        self.historico_elem[el][passo_global] = {
                            'SXX': m_S[0], 'SYY': m_S[1], 'SZZ': m_S[2], 'SXZ': m_S[3],
                            'S1': m_S[4], 'S3': m_S[5], 'ANGLE': m_S[7], 'PWP': pwp, 'RM': m_S[10],
                            'EXX': m_E[0], 'EYY': m_E[1], 'EZZ': m_E[2], 'EXZ': m_E[3],
                            'E1': m_E[4], 'E3': m_E[5], 'EV': ev,
                            'S_DEV': m_S[4] - m_S[5], 'E_DEV': m_E[4] - m_E[5],
                            'SXX_TOT': m_S[0] - pwp, 'SYY_TOT': m_S[1] - pwp,
                            'SZZ_TOT': m_S[2] - pwp, 'S1_TOT': m_S[4] - pwp,
                            'S3_TOT': m_S[5] - pwp,
                            'GAUSS': gauss_vals
                        }
                    continue

                if lendo_nos_result:
                    if "LARGEST INDIVIDUAL RESIDUAL" in linha or linha_strip == "":
                        lendo_nos_result = False
                    else:
                        partes = linha.split()
                        if len(partes) >= 7 and partes[0].isdigit():
                            try:
                                id_no = int(partes[0])
                                self.historico_nos.setdefault(id_no, {})[passo_global] = {
                                    'dX': float(partes[5]),
                                    'dZ': float(partes[6])
                                }
                            except ValueError:
                                pass

        self.lista_materiais = sorted(set(self.materiais.values()))

    def _calcular_escalas_base(self):
        x_coords = [n['X'] for n in self.nos.values()]
        self.largura_barragem = max(x_coords) - min(x_coords) if x_coords else 100.0
        max_d = max(
            [np.hypot(p['dX'], p['dZ']) for no in self.historico_nos.values() for p in no.values()] + [0.0]
        )
        self.max_desloc_global = max_d if max_d > 0 else 0.001

    @staticmethod
    def _shape8(r, s):
        return np.array([
            -0.25 * (1-r) * (1-s) * (1+r+s),
             0.50 * (1-r*r) * (1-s),
            -0.25 * (1+r) * (1-s) * (1-r+s),
             0.50 * (1+r) * (1-s*s),
            -0.25 * (1+r) * (1+s) * (1-r-s),
             0.50 * (1-r*r) * (1+s),
            -0.25 * (1-r) * (1+s) * (1+r-s),
             0.50 * (1-r) * (1-s*s),
        ])

    def gerar_triangulacao_ativa(self, passo, materiais_ativos, forcar_tudo=False):
        node_ids = sorted(self.nos.keys())
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        x = [self.nos[nid]['X'] for nid in node_ids]
        z = [self.nos[nid]['Z'] for nid in node_ids]
        triangulos, mask = [], []
        nos_ativos = set()

        for el, conec in self.elementos.items():
            ativo = True if forcar_tudo else (
                passo in self.historico_elem.get(el, {}) and
                self.materiais.get(el) in materiais_ativos
            )
            if len(conec) != 8 or any(n not in node_to_idx for n in conec):
                continue
            ids = [node_to_idx[conec[i]] for i in range(8)]
            triangulos.extend([
                [ids[0], ids[1], ids[7]], [ids[1], ids[2], ids[3]], [ids[3], ids[4], ids[5]],
                [ids[5], ids[6], ids[7]], [ids[1], ids[3], ids[7]], [ids[3], ids[5], ids[7]]
            ])
            mask.extend([not ativo] * 6)
            if ativo:
                nos_ativos.update(conec)

        if not triangulos:
            # Triangulação mínima válida para manter a interface robusta.
            return None, node_to_idx, nos_ativos

        triang = mtri.Triangulation(x, z, triangulos)
        triang.set_mask(mask)
        return triang, node_to_idx, nos_ativos

    def _gauss_ordenado(self, passo, el, variavel):
        hist = self.historico_elem.get(el, {})
        if passo not in hist:
            return None
        gauss = hist[passo].get('GAUSS', {})
        if variavel not in gauss:
            return None
        g_file = np.asarray(gauss[variavel], dtype=float).reshape(-1)
        if g_file.size < 4:
            return None
        return np.array([g_file[0], g_file[2], g_file[3], g_file[1]], dtype=float)

    def preparar_dados_campo(self, passo, variavel, materiais_ativos):
        dados = []
        for el, conec in self.elementos.items():
            if self.materiais.get(el) not in materiais_ativos:
                continue
            if passo not in self.historico_elem.get(el, {}):
                continue
            g = self._gauss_ordenado(passo, el, variavel)
            if g is None or np.count_nonzero(np.isfinite(g)) < 3:
                continue
            try:
                xy = tuple(
                    float(v)
                    for n in conec
                    for v in (self.nos[n]['X'], self.nos[n]['Z'])
                )
            except KeyError:
                continue
            dados.append((int(el), int(self.materiais[el]), tuple(conec), xy, tuple(float(v) for v in g)))
        return tuple(dados)

    def limites_gauss(self, passo, variavel, materiais_ativos):
        dados = self.preparar_dados_campo(passo, variavel, materiais_ativos)
        vals = [v for item in dados for v in item[4] if np.isfinite(v)]
        if not vals:
            return 0.0, 1.0
        v_min, v_max = float(np.min(vals)), float(np.max(vals))
        if variavel == 'RM':
            v_min = min(0.0, v_min)
        if np.isclose(v_min, v_max):
            delta = max(abs(v_min) * 0.05, 1e-6)
            v_min -= delta
            v_max += delta
        return v_min, v_max


    def _interpolar_para_nos(self, passo, variavel):
        valores_nos = np.zeros(len(self.nos), dtype=float)
        contagem = np.zeros(len(self.nos), dtype=float)
        node_to_idx = {nid: i for i, nid in enumerate(sorted(self.nos.keys()))}
        a = 1.0 + np.sqrt(3.0)/2.0
        b = -0.5
        c = 1.0 - np.sqrt(3.0)/2.0

        for el, conec in self.elementos.items():
            hist = self.historico_elem.get(el, {})
            if passo not in hist:
                continue
            g = self._gauss_ordenado(passo, el, variavel)
            if g is not None:
                g1, g2, g3, g4 = g
                v_corners = [
                    a*g1 + b*g2 + b*g3 + c*g4,
                    b*g1 + c*g2 + a*g3 + b*g4,
                    c*g1 + b*g2 + b*g3 + a*g4,
                    b*g1 + a*g2 + c*g3 + b*g4,
                ]
                for i, no in enumerate(conec):
                    idx = node_to_idx[no]
                    if i == 0: val = v_corners[0]
                    elif i == 2: val = v_corners[1]
                    elif i == 4: val = v_corners[2]
                    elif i == 6: val = v_corners[3]
                    elif i == 1: val = 0.5*(v_corners[0]+v_corners[1])
                    elif i == 3: val = 0.5*(v_corners[1]+v_corners[2])
                    elif i == 5: val = 0.5*(v_corners[2]+v_corners[3])
                    else: val = 0.5*(v_corners[3]+v_corners[0])
                    valores_nos[idx] += val
                    contagem[idx] += 1.0
            else:
                val = hist[passo].get(variavel, np.nan)
                if np.isfinite(val):
                    for no in conec:
                        idx = node_to_idx[no]
                        valores_nos[idx] += val
                        contagem[idx] += 1.0

        with np.errstate(invalid='ignore', divide='ignore'):
            valores_nos = np.divide(valores_nos, contagem, out=np.zeros_like(valores_nos), where=contagem > 0)
        return np.nan_to_num(valores_nos)

    def malha_campo(self, passo, variavel, materiais_ativos, n_local=20):
        dados = self.preparar_dados_campo(passo, variavel, materiais_ativos)
        nx = min(900, max(420, int(n_local * 36)))
        nz = min(600, max(280, int(n_local * 24)))
        malha = reconstruir_malha_global_cache(dados, nx=nx, nz=nz, sigma=0.0)
        grade = grade_cache_para_mpl(malha)
        return grade, dados

    def limites_campo(self, passo, variavel, materiais_ativos):
        dados = self.preparar_dados_campo(passo, variavel, materiais_ativos)
        vals = [v for item in dados for v in item[4] if np.isfinite(v)]
        if not vals:
            return 0.0, 1.0
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if variavel == 'RM':
            vmin = min(0.0, vmin)
        if np.isclose(vmin, vmax):
            d = max(abs(vmax)*0.05, 1e-6)
            vmin -= d; vmax += d
        return vmin, vmax

    def valor_elemento(self, el, passo, variavel):
        h = self.historico_elem.get(int(el), {})
        if passo not in h:
            return np.nan
        return float(h[passo].get(variavel, np.nan))

    def serie_elemento(self, el, variavel):
        h = self.historico_elem.get(int(el), {})
        passos = sorted(h.keys())
        return passos, [self.valor_elemento(el, p, variavel) for p in passos]




st.set_page_config(page_title="Pós-Processador PROGEO", layout="wide")

@st.cache_resource(show_spinner=False)
def carregar_modelo(file_bytes):
    with open("temp.pri", "wb") as f:
        f.write(file_bytes)
    return LeitorPROGEO("temp.pri")

st.title("Pós-Processador PROGEO")
st.info(
    "🛠️ **Desenvolvido por:** Victor Cavalcanti "
    "— *Engenheiro Civil | Mestrando em Geotecnia (COPPE/UFRJ)*\n\n"
    "Ferramenta de pós-processamento de dados do PROGEO. Seu uso não elimina a necessidade de utilizar o pós-processador oficial (Postgeo), ou o software integrado Sysgeo."
    " Contato: victor.cavalcanti@coc.ufrj.br"
)

uploaded_file = st.file_uploader("Faça o upload do seu arquivo .PRI", type=["pri"])

if uploaded_file is not None:
    progeo = carregar_modelo(uploaded_file.getvalue())
    escala_base_visual = (progeo.largura_barragem * 0.05) / progeo.max_desloc_global

    st.sidebar.header("Controles da Malha")
    passo = st.sidebar.slider("Passo Global", 1, progeo.total_passos, progeo.total_passos)

    mapa_variaveis = {
        "Apenas Geometria": "Geometria Base",
        "🔸 TENSÕES": "Geometria Base",
        "    Tensão Vertical (σz)": "SZZ",
        "    Tensão Horizontal (σx)": "SXX",
        "    Tensão Principal Maior (σ1)": "S1",
        "    Tensão Principal Menor (σ3)": "S3",
        "    Tensão Desviadora (q)": "S_DEV",
        "    Tensão Cisalhante (τxz)": "SXZ",
        "    Poropressão (u)": "PWP",
        "🔸 DEFORMAÇÕES": "Geometria Base",
        "    Deformação Vertical (εz)": "EZZ",
        "    Deformação Horizontal (εx)": "EXX",
        "    Deformação Principal Maior (ε1)": "E1",
        "    Deformação Principal Menor (ε3)": "E3",
        "    Deformação Cisalhante (γ)": "E_DEV",
        "    Deformação Volumétrica (εv)": "EV",
        "🔸 PLASTIFICAÇÃO": "Geometria Base",
        "    Resistência Mobilizada (R)": "RM",
    }

    opcao_selecionada = st.sidebar.selectbox("Campo", list(mapa_variaveis.keys()), index=0)
    variavel = mapa_variaveis[opcao_selecionada]
    mats_ativos = st.sidebar.multiselect("Materiais Ativos", progeo.lista_materiais, default=progeo.lista_materiais)

    st.sidebar.markdown("---")
    ver_def = st.sidebar.checkbox("Rede Deformada")
    mult_def = st.sidebar.number_input("Escala Deformada", value=1.00, step=0.1)
    ver_vet = st.sidebar.checkbox("Vetores de Deslocamento")
    mult_vet = st.sidebar.number_input("Escala Vetor", value=1.00, step=0.1)
    ver_cruz = st.sidebar.checkbox("Cruzes de Tensão")
    esc_cruz = st.sidebar.number_input("Escala Cruz", value=0.005, step=0.001, format="%.3f")
    ver_id_nos = st.sidebar.checkbox("IDs dos Nós")
    ver_id_el = st.sidebar.checkbox("IDs dos Elementos")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Câmera")
    x_orig_all = [progeo.nos[n]['X'] for n in progeo.nos]
    z_orig_all = [progeo.nos[n]['Z'] for n in progeo.nos]
    col_x1, col_x2 = st.sidebar.columns(2)
    zoom_x_min = col_x1.number_input("X Mín", value=float(min(x_orig_all)))
    zoom_x_max = col_x2.number_input("X Máx", value=float(max(x_orig_all)))
    col_z1, col_z2 = st.sidebar.columns(2)
    zoom_z_min = col_z1.number_input("Z Mín", value=float(min(z_orig_all)))
    zoom_z_max = col_z2.number_input("Z Máx", value=float(max(z_orig_all)))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎞️ Animação")
    anim_ini, anim_fim = st.sidebar.columns(2)
    passo_ini_anim = anim_ini.number_input("Inicial", 1, progeo.total_passos, max(1, passo-10))
    passo_fim_anim = anim_fim.number_input("Final", 1, progeo.total_passos, passo)
    vel_anim = st.sidebar.slider("Velocidade (s/frame)", 0.05, 1.00, 0.20, 0.05)
    iniciar_animacao = st.sidebar.button("▶ Animar passos")
    exportar_gif = st.sidebar.button("🎞️ Exportar GIF")

    aba_malha, aba_graficos = st.tabs(["Visualização 2D (Malha)", "Gráficos Analíticos"])

    def adicionar_info_passo(ax, passo_local, variavel_local):
        info_local = progeo.passo_info.get(passo_local, {'Estagio':'-','Inc':'-'})
        texto = f"Passo Global: {passo_local}  |  Estágio: {info_local['Estagio']}  |  Incremento: {info_local['Inc']}"
        ax.text(0.5, 1.015, texto, transform=ax.transAxes, ha='center', va='bottom', fontsize=9, color='black')
        rotulo = 'ratio  R' if variavel_local == 'RM' else variavel_local
        ax.text(0.98, -0.09, rotulo, transform=ax.transAxes, ha='right', va='top', fontsize=8, color='black')

    def excel_bytes(worksheets):
        """Converte DataFrames em um arquivo XLSX em memória."""
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            for nome, dataframe in worksheets.items():
                if dataframe is not None and not dataframe.empty:
                    dataframe.to_excel(writer, sheet_name=str(nome)[:31], index=False)
        buffer.seek(0)
        return buffer.getvalue()

    def gerar_gif_mapa(p0, p1, levels_fixos):
        """Gera GIF da sequência de passos usando as mesmas cores/escala."""
        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError("Para exportar GIF, instale Pillow: python -m pip install pillow")
        frames = []
        p0, p1 = int(min(p0, p1)), int(max(p0, p1))
        passos_validos = [p for p in range(p0, p1 + 1) if p in progeo.passo_info]
        progress = st.progress(0.0, text="Gerando GIF…")
        for i, pf in enumerate(passos_validos):
            fig_f, ax_f = plt.subplots(figsize=(10, 5.7), constrained_layout=True)
            desenhar_mapa(fig_f, ax_f, pf, levels_override=levels_fixos, n_local_visual=11, para_gif=True)
            bio = BytesIO()
            fig_f.savefig(bio, format='png', dpi=120, bbox_inches='tight', facecolor='white')
            plt.close(fig_f)
            bio.seek(0)
            frames.append(Image.open(bio).convert('P', palette=Image.ADAPTIVE, colors=256))
            progress.progress((i + 1) / max(1, len(passos_validos)), text=f"Gerando GIF… passo {pf}/{p1}")
        progress.empty()
        if not frames:
            return None
        out = BytesIO()
        duration_ms = max(60, int(float(vel_anim) * 1000.0))
        frames[0].save(out, format='GIF', save_all=True, append_images=frames[1:],
                       duration=duration_ms, loop=0, optimize=False, disposal=2)
        out.seek(0)
        return out.getvalue()

    def desenhar_mapa(fig, ax, passo_local, levels_override=None, n_local_visual=20, para_gif=False):
        triang_fundo, _, _ = progeo.gerar_triangulacao_ativa(passo_local, mats_ativos, forcar_tudo=True)
        triang_ativo, map_nos, nos_ativos = progeo.gerar_triangulacao_ativa(passo_local, mats_ativos)

        if variavel == 'Geometria Base':
            if triang_fundo is not None:
                ax.triplot(triang_fundo, color='lightgray', linewidth=0.35, alpha=0.45)
            if triang_ativo is not None:
                ax.triplot(triang_ativo, color='gray', linewidth=0.55, alpha=0.55)
        elif not mats_ativos:
            ax.text(0.5, 0.5, "Nenhum material ativo.", ha='center', va='center', transform=ax.transAxes, color='red')
        else:
            grade, _ = progeo.malha_campo(passo_local, variavel, mats_ativos, n_local=n_local_visual)
            if grade is None:
                ax.text(0.5, 0.5, "Nenhum resultado disponível para os materiais/ passo selecionados.", ha='center', va='center', transform=ax.transAxes, color='red')
            else:
                GX, GZ, G = grade
                if levels_override is None:
                    vmin, vmax = progeo.limites_campo(passo_local, variavel, mats_ativos)
                    niveis = np.linspace(vmin, vmax, 11)
                else:
                    niveis = np.asarray(levels_override, dtype=float)
                contorno = ax.contourf(GX, GZ, G, levels=niveis, cmap='jet', antialiased=True)
                try:
                    ax.contour(GX, GZ, G, levels=niveis, colors='k', linewidths=0.18, alpha=0.22)
                except Exception:
                    pass
                cbar = fig.colorbar(contorno, ax=ax, ticks=niveis, format='%.2e', pad=0.015)
                cbar.set_label('ratio   R' if variavel == 'RM' else variavel, size=8)
                cbar.ax.tick_params(labelsize=7)

        if ver_def and triang_ativo is not None:
            fator_def = escala_base_visual * mult_def
            node_ids = sorted(progeo.nos.keys())
            x_def = [progeo.nos[n]['X'] + progeo.historico_nos[n].get(passo_local, {'dX':0.0})['dX']*fator_def for n in node_ids]
            z_def = [progeo.nos[n]['Z'] + progeo.historico_nos[n].get(passo_local, {'dZ':0.0})['dZ']*fator_def for n in node_ids]
            tri_def = mtri.Triangulation(x_def, z_def, triang_ativo.triangles)
            tri_def.set_mask(triang_ativo.mask)
            ax.triplot(tri_def, color='green', linewidth=0.6, alpha=0.6)

        if ver_vet and nos_ativos:
            fator_vet = escala_base_visual * mult_vet
            X_vet, Z_vet, U_vet, V_vet = [], [], [], []
            for n in sorted(nos_ativos):
                d = progeo.historico_nos[n].get(passo_local, {'dX':0.0, 'dZ':0.0})
                X_vet.append(progeo.nos[n]['X']); Z_vet.append(progeo.nos[n]['Z'])
                U_vet.append(d['dX'] * fator_vet); V_vet.append(d['dZ'] * fator_vet)
            ax.quiver(X_vet, Z_vet, U_vet, V_vet, color='darkmagenta', angles='xy', scale_units='xy', scale=1, width=0.0028, zorder=5)

        for el, hist in progeo.historico_elem.items():
            if passo_local not in hist or progeo.materiais.get(el) not in mats_ativos:
                continue
            ns = progeo.elementos[el]
            cantos = [ns[0], ns[2], ns[4], ns[6]]
            xc = np.mean([progeo.nos[n]['X'] for n in cantos]); zc = np.mean([progeo.nos[n]['Z'] for n in cantos])
            if ver_cruz:
                s1, s3 = hist[passo_local]['S1']*esc_cruz, hist[passo_local]['S3']*esc_cruz
                ang = np.deg2rad(hist[passo_local].get('ANGLE',0))
                dx1, dz1 = s1*np.cos(ang), s1*np.sin(ang)
                dx3, dz3 = s3*np.cos(ang+np.pi/2), s3*np.sin(ang+np.pi/2)
                ax.plot([xc-dx1,xc+dx1],[zc-dz1,zc+dz1], color='red' if hist[passo_local]['S1']<0 else 'blue', linewidth=0.5)
                ax.plot([xc-dx3,xc+dx3],[zc-dz3,zc+dz3], color='red' if hist[passo_local]['S3']<0 else 'blue', linewidth=0.5)
            if ver_id_el:
                ax.text(xc, zc, str(el), fontsize=4, color='maroon', weight='bold', ha='center', va='center')

        if ver_id_nos:
            for n_id, nd in progeo.nos.items():
                if n_id in nos_ativos:
                    ax.text(nd['X'], nd['Z'], str(n_id), fontsize=3, color='black', ha='center', va='center', zorder=10)

        info = progeo.passo_info.get(passo_local, {'Estagio':'-','Inc':'-'})
        ax.set_xlim(zoom_x_min, zoom_x_max); ax.set_ylim(zoom_z_min, zoom_z_max)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X (m)', fontsize=7)
        ax.set_ylabel('Z (m)', fontsize=7)
        ax.tick_params(axis='both', labelsize=6)
        ax.grid(True, linestyle=':', alpha=0.35)
        adicionar_info_passo(ax, passo_local, variavel)

    with aba_malha:
        info_atual = progeo.passo_info.get(passo, {'Estagio':'-','Inc':'-'})
        st.caption(f"**Passo Global {passo}**  |  **Estágio {info_atual['Estagio']}**  |  **Incremento {info_atual['Inc']}**  |  Materiais ativos: {', '.join(map(str, mats_ativos)) if mats_ativos else 'nenhum'}")
        if exportar_gif and variavel != 'Geometria Base' and mats_ativos:
            try:
                p0_g, p1_g = int(min(passo_ini_anim, passo_fim_anim)), int(max(passo_ini_anim, passo_fim_anim))
                anim_vals = []
                for pp in range(p0_g, p1_g + 1):
                    dpp = progeo.preparar_dados_campo(pp, variavel, mats_ativos)
                    anim_vals.extend([v for item in dpp for v in item[4] if np.isfinite(v)])
                if anim_vals:
                    avmin, avmax = float(np.min(anim_vals)), float(np.max(anim_vals))
                    if variavel == 'RM':
                        avmin = min(0.0, avmin)
                    if np.isclose(avmin, avmax):
                        dd = max(abs(avmax) * 0.05, 1e-6); avmin -= dd; avmax += dd
                    gif_levels = np.linspace(avmin, avmax, 11)
                    gif_bytes = gerar_gif_mapa(p0_g, p1_g, gif_levels)
                    if gif_bytes:
                        st.success(f"GIF gerado: passos {p0_g}–{p1_g}.")
                        st.download_button("📥 Baixar animação GIF", gif_bytes, "animacao_progeo.gif", "image/gif", key="download_gif")
                else:
                    st.warning("Não há dados suficientes para gerar o GIF no intervalo selecionado.")
            except Exception as exc:
                st.error(f"Não foi possível gerar o GIF: {exc}")

        if iniciar_animacao and variavel != 'Geometria Base' and mats_ativos:
            placeholder = st.empty()
            p0, p1 = int(min(passo_ini_anim, passo_fim_anim)), int(max(passo_ini_anim, passo_fim_anim))
            import time
            # Escala fixa durante a animação: evita que as cores mudem de significado
            # a cada passo. Os limites são obtidos diretamente dos Gauss, sem reconstrução.
            if variavel != 'Geometria Base' and mats_ativos:
                anim_vals=[]
                for pp in range(p0, p1+1):
                    dpp=progeo.preparar_dados_campo(pp, variavel, mats_ativos)
                    anim_vals.extend([v for item in dpp for v in item[4] if np.isfinite(v)])
                if anim_vals:
                    avmin,avmax=float(np.min(anim_vals)),float(np.max(anim_vals))
                    if variavel=='RM': avmin=min(0.0,avmin)
                    if np.isclose(avmin,avmax):
                        dd=max(abs(avmax)*.05,1e-6); avmin-=dd; avmax+=dd
                    anim_levels=np.linspace(avmin,avmax,11)
                else:
                    anim_levels=None
            else:
                anim_levels=None
            for p_anim in range(p0, p1 + 1):
                fig_anim, ax_anim = plt.subplots(figsize=(10, 5.7), constrained_layout=True)
                desenhar_mapa(fig_anim, ax_anim, p_anim, levels_override=anim_levels, n_local_visual=11)
                placeholder.pyplot(fig_anim)
                plt.close(fig_anim)
                time.sleep(float(vel_anim))
        else:
            fig_malha, ax_malha = plt.subplots(figsize=(10, 5.7), constrained_layout=True)
            desenhar_mapa(fig_malha, ax_malha, passo)
            st.pyplot(fig_malha)
            plt.close(fig_malha)

    with aba_graficos:
        st.subheader("Ferramentas de Análise")
        modo = st.selectbox("Análise", [
            'Trajetória p-q (elemento)',
            'Comparação p-q entre elementos',
            'Perfis de evolução ao longo de uma seção',
            'Variável × Passo Global',
            'Valores em Seção de Reta'
        ])

        if modo == 'Trajetória p-q (elemento)':
            elid = st.number_input("Elemento", min_value=1, value=1, step=1)
            teoria = st.selectbox("Formulação", ['MIT','Cambridge'])
            env = st.checkbox("Plotar Envoltória", value=True)
            c_linha = st.number_input("Coesão c'", value=0.0)
            phi_linha = st.number_input("Ângulo φ'", value=0.0)
            if st.button("Gerar trajetória"):
                fig, ax = plt.subplots(figsize=(10,5)); df_export={}
                if elid in progeo.historico_elem:
                    passos_el, pvals, qvals = sorted(progeo.historico_elem[elid].keys()), [], []
                    for pp in passos_el:
                        h=progeo.historico_elem[elid][pp]; s1,s3,syy=h['S1'],h['S3'],h['SYY']
                        if teoria=='MIT': pvals.append(-(s1+s3)/2.0); qvals.append(abs(s1-s3)/2.0)
                        else: pvals.append(-(s1+syy+s3)/3.0); qvals.append((1/np.sqrt(2))*np.sqrt((s1-syy)**2+(syy-s3)**2+(s3-s1)**2))
                    ax.plot(pvals,qvals,'-o',label=f'Elemento {elid}')
                    if env:
                        ph=np.deg2rad(phi_linha); pl=np.linspace(0,max(pvals)*1.2 if pvals else 100,100)
                        if teoria=='MIT': ql=pl*np.sin(ph)+c_linha*np.cos(ph)
                        else: M=(6*np.sin(ph))/(3-np.sin(ph)); ql=M*pl+c_linha*(6*np.cos(ph))/(3-np.sin(ph))
                        ax.plot(pl,ql,'--',label='Envoltória')
                    ax.set_xlabel("p'"); ax.set_ylabel('q'); ax.set_title(f'Trajetória p-q — Elemento {elid}'); ax.legend(); ax.grid(True,alpha=0.3)
                    df_export={'Passo_Global':pd.Series(passos_el),'P_Efetivo':pd.Series(pvals),'Q_Desviador':pd.Series(qvals)}
                st.pyplot(fig); plt.close(fig)
                df_pq = pd.DataFrame(df_export)
                st.download_button('📥 Baixar CSV', df_pq.to_csv(index=False).encode('utf-8'), 'trajetoria_pq.csv','text/csv', key='csv_pq_single')
                if not df_pq.empty:
                    st.download_button('📊 Baixar Excel', excel_bytes({'Trajetoria_pq': df_pq}), 'trajetoria_pq.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='xlsx_pq_single')

        elif modo == 'Comparação p-q entre elementos':
            ids_disp=sorted(progeo.elementos.keys())
            els = st.multiselect("Elementos", ids_disp, default=ids_disp[:2] if len(ids_disp)>=2 else ids_disp)
            teoria = st.selectbox("Formulação", ['MIT','Cambridge'], key='pq_cmp')
            if st.button("Comparar elementos"):
                fig,ax=plt.subplots(figsize=(10,5)); registros=[]
                for elid in els:
                    if elid not in progeo.historico_elem: continue
                    pvals, qvals = [], []
                    for ps in sorted(progeo.historico_elem[elid].keys()):
                        h=progeo.historico_elem[elid][ps]; s1,s3,syy=h['S1'],h['S3'],h['SYY']
                        if teoria=='MIT': pval=-(s1+s3)/2.0; qval=abs(s1-s3)/2.0
                        else: pval=-(s1+syy+s3)/3.0; qval=(1/np.sqrt(2))*np.sqrt((s1-syy)**2+(syy-s3)**2+(s3-s1)**2)
                        pvals.append(pval); qvals.append(qval); registros.append({'Elemento':elid,'Passo_Global':ps,'P_Efetivo':pval,'Q_Desviador':qval})
                    ax.plot(pvals,qvals,'-o',markersize=3,label=f'Elem. {elid}')
                ax.set_xlabel("p'"); ax.set_ylabel('q'); ax.set_title('Comparação de trajetórias p-q'); ax.grid(True,alpha=0.3); ax.legend(); st.pyplot(fig); plt.close(fig)
                df_cmp=pd.DataFrame(registros)
                if not df_cmp.empty:
                    st.download_button('📥 Baixar CSV', df_cmp.to_csv(index=False).encode('utf-8'), 'comparacao_pq.csv','text/csv', key='csv_pq_cmp')
                    st.download_button('📊 Baixar Excel', excel_bytes({'Comparacao_pq': df_cmp}), 'comparacao_pq.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='xlsx_pq_cmp')

        elif modo == 'Perfis de evolução ao longo de uma seção':
            var_prof = st.selectbox("Variável", [k for k in mapa_variaveis.values() if k!='Geometria Base'], index=0)
            col1,col2,col3,col4 = st.columns(4)
            x0p=col1.number_input('X inicial', value=0.0, key='px0'); z0p=col2.number_input('Z inicial', value=0.0, key='pz0')
            x1p=col3.number_input('X final', value=10.0, key='px1'); z1p=col4.number_input('Z final', value=5.0, key='pz1')
            passos_sel=st.multiselect('Passos para comparar', list(range(1,progeo.total_passos+1)), default=[max(1,progeo.total_passos//3), max(1,2*progeo.total_passos//3), progeo.total_passos])
            if st.button('Gerar perfis'):
                fig,ax=plt.subplots(figsize=(10,5));
                df={}
                dist=np.linspace(0,1,150)
                xlin=x0p+(x1p-x0p)*dist; zlin=z0p+(z1p-z0p)*dist
                df['Distancia'] = np.sqrt((xlin-x0p)**2+(zlin-z0p)**2)
                for pprof in sorted(passos_sel):
                    try:
                        grade,_ = progeo.malha_campo(pprof,var_prof,mats_ativos,n_local=14)
                        if grade is None: continue
                        GX,GZ,GV=grade
                        y=interpolar_grade_regular(GX,GZ,GV,xlin,zlin)
                        ax.plot(df['Distancia'],y,label=f'Passo {pprof}')
                        df[f'Passo_{pprof}']=y
                    except Exception: continue
                ax.set_xlabel('Distância ao longo da seção'); ax.set_ylabel(var_prof); ax.set_title(f'Perfil de evolução — {var_prof}'); ax.grid(True,alpha=0.3); ax.legend(); st.pyplot(fig); plt.close(fig)
                df_prof=pd.DataFrame(df)
                st.download_button('📥 Baixar perfis CSV', df_prof.to_csv(index=False).encode('utf-8'),'perfis_evolucao.csv','text/csv', key='csv_profiles')
                if not df_prof.empty:
                    st.download_button('📊 Baixar Excel', excel_bytes({'Perfis_evolucao': df_prof}), 'perfis_evolucao.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='xlsx_profiles')

        elif modo == 'Variável × Passo Global':
            tipo_alvo=st.radio('Fonte do histórico', ['Elemento','Nó'], horizontal=True)
            if tipo_alvo=='Elemento':
                var_hist=st.selectbox('Variável', ['PWP','RM','SZZ','SXX','S1','S3','S_DEV','SXZ','EZZ','EXX','E1','E3','EV'])
                elhist=st.number_input('Elemento', min_value=1, value=1, step=1)
                if st.button('Gerar histórico'):
                    if elhist in progeo.historico_elem:
                        ps,ys=progeo.serie_elemento(elhist,var_hist)
                        fig,ax=plt.subplots(figsize=(10,5)); ax.plot(ps,ys,'-o',markersize=3); ax.set_xlabel('Passo Global'); ax.set_ylabel(var_hist); ax.set_title(f'{var_hist} × Passo — Elemento {elhist}'); ax.grid(True,alpha=0.3); st.pyplot(fig); plt.close(fig)
                        df_hist=pd.DataFrame({'Passo_Global':ps,var_hist:ys})
                        st.download_button('📥 Baixar série CSV', df_hist.to_csv(index=False).encode('utf-8'),'historico_variavel.csv','text/csv', key='csv_hist_elem')
                        st.download_button('📊 Baixar Excel', excel_bytes({'Historico_elemento': df_hist}), 'historico_variavel.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='xlsx_hist_elem')
                    else: st.warning('Elemento não encontrado.')
            else:
                var_no=st.selectbox('Variável nodal', ['dZ','dX','|d|'])
                nohist=st.number_input('Nó', min_value=1, value=1, step=1)
                if st.button('Gerar histórico nodal'):
                    if nohist in progeo.historico_nos:
                        ps=sorted(progeo.historico_nos[nohist].keys())
                        vals=[]
                        for pp in ps:
                            d=progeo.historico_nos[nohist][pp]
                            vals.append(float(np.hypot(d['dX'],d['dZ'])) if var_no=='|d|' else float(d[var_no]))
                        fig,ax=plt.subplots(figsize=(10,5)); ax.plot(ps,vals,'-o',markersize=3); ax.set_xlabel('Passo Global'); ax.set_ylabel(var_no); ax.set_title(f'{var_no} × Passo — Nó {nohist}'); ax.grid(True,alpha=0.3); st.pyplot(fig); plt.close(fig)
                        df_hist_no=pd.DataFrame({'Passo_Global':ps,var_no:vals})
                        st.download_button('📥 Baixar série nodal CSV', df_hist_no.to_csv(index=False).encode('utf-8'),'historico_nodal.csv','text/csv', key='csv_hist_node')
                        st.download_button('📊 Baixar Excel', excel_bytes({'Historico_nodal': df_hist_no}), 'historico_nodal.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='xlsx_hist_node')
                    else: st.warning('Nó não encontrado.')

        elif modo == 'Valores em Seção de Reta':
            var_sec = st.selectbox('Variável da seção', [k for k in mapa_variaveis.values() if k!='Geometria Base'], index=0)
            c1,c2,c3,c4=st.columns(4)
            xs0=c1.number_input('X início', value=0.0, key='sx0'); zs0=c2.number_input('Z início', value=0.0, key='sz0')
            xs1=c3.number_input('X final', value=10.0, key='sx1'); zs1=c4.number_input('Z final', value=5.0, key='sz1')
            if st.button('Gerar seção'):
                fig,ax=plt.subplots(figsize=(10,5)); xlin=np.linspace(xs0,xs1,150); zlin=np.linspace(zs0,zs1,150)
                try:
                    grade,_=progeo.malha_campo(passo,var_sec,mats_ativos,n_local=14)
                    if grade is not None:
                        GX,GZ,GV=grade; y=interpolar_grade_regular(GX,GZ,GV,xlin,zlin); dist=np.sqrt((xlin-xs0)**2+(zlin-zs0)**2)
                        ax.plot(dist,y,linewidth=2); ax.set_xlabel('Distância (m)'); ax.set_ylabel(var_sec); ax.set_title(f'{var_sec} ao longo da linha — Passo {passo}'); ax.grid(True,alpha=0.3); st.pyplot(fig); plt.close(fig)
                        df_sec=pd.DataFrame({'Distancia_m':dist,'X_coord':xlin,'Z_coord':zlin,'Valor':y})
                        st.download_button('📥 Baixar seção CSV', df_sec.to_csv(index=False).encode('utf-8'),'secao_resultado.csv','text/csv', key='csv_section')
                        st.download_button('📊 Baixar Excel', excel_bytes({'Secao_resultado': df_sec}), 'secao_resultado.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='xlsx_section')
                    else: st.warning('Não há campo disponível para esta seleção.')
                except Exception as exc: st.warning(f'Não foi possível gerar a seção: {exc}')
