import streamlit as st
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# ==============================================================================
# 1. PARSER GEOTÉCNICO PROGEO (Core Matemático)
# ==============================================================================
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
        
        self.regex_float = r'-?\d*\.\d+(?:E[+-]\d+)?|-?\d+\.\d+'
        self._parse_arquivo()
        self._calcular_escalas_base()

    def _parse_arquivo(self):
        estagio_atual = 0
        passo_global = 0
        lendo_coord = lendo_elem = lendo_nos_result = False
        gauss_temp_S, gauss_temp_E = {}, {}

        with open(self.caminho, 'r', encoding='latin-1', errors='replace') as f:
            for linha in f:
                linha_strip = linha.strip()
                
                if "NODE     X CO-ORD     Z CO-ORD" in linha:
                    lendo_coord = True
                    continue
                if lendo_coord:
                    if linha_strip == "" or "BOUNDARY" in linha: lendo_coord = False
                    else:
                        partes = linha.split()
                        if len(partes) >= 3 and partes[0].isdigit():
                            self.nos[int(partes[0])] = {'X': float(partes[1]), 'Z': float(partes[2])}
                            self.historico_nos[int(partes[0])] = {}

                if "ELEMENT           C O N N E C T I O N S" in linha:
                    lendo_elem = True
                    continue
                if lendo_elem:
                    if "NO. OF ELEMENTS" in linha or linha_strip == "": lendo_elem = False
                    else:
                        partes = linha.split()
                        if len(partes) >= 11 and partes[0].isdigit():
                            id_el = int(partes[0])
                            self.elementos[id_el] = [int(p) for p in partes[1:9]]
                            self.materiais[id_el] = int(partes[10])
                            if id_el not in self.historico_elem:
                                self.historico_elem[id_el] = {}

                match_inc = re.search(r'START OF INCREMENT NO\.\s+(\d+)', linha)
                if match_inc:
                    incremento_local = int(match_inc.group(1))
                    if incremento_local == 1: estagio_atual += 1 
                    
                    passo_global += 1
                    self.total_passos = passo_global
                    self.passo_info[passo_global] = {'Estagio': estagio_atual, 'Inc': incremento_local}
                    
                    gauss_temp_S = {el: [] for el in self.elementos.keys()}
                    gauss_temp_E = {el: [] for el in self.elementos.keys()}

                match_el = re.search(r'EL\.NO\.\s+(\d+)', linha)
                if match_el: el_atual = int(match_el.group(1))
                if " S= " in linha and el_atual is not None:
                    vals = re.findall(self.regex_float, linha)
                    if len(vals) >= 12: gauss_temp_S[el_atual].append([float(v) for v in vals[1:12]])
                if " E= " in linha and el_atual is not None:
                    vals = re.findall(self.regex_float, linha)
                    if len(vals) >= 9: gauss_temp_E[el_atual].append([float(v) for v in vals[1:9]])

                if "TOTAL NODAL VALUES" in linha:
                    lendo_nos_result = True
                    for el in self.elementos.keys():
                        if gauss_temp_S[el]:
                            m_S = np.mean(gauss_temp_S[el], axis=0)
                            m_E = np.mean(gauss_temp_E[el], axis=0)
                            self.historico_elem[el][passo_global] = {
                                'SXX': m_S[0], 'SYY': m_S[1], 'SZZ': m_S[2], 'SXZ': m_S[3],
                                'S1': m_S[4], 'S3': m_S[5], 'ANGLE': m_S[7], 'PWP': m_S[9], 'RM': m_S[10],
                                'EXX': m_E[0], 'EYY': m_E[1], 'EZZ': m_E[2], 'EXZ': m_E[3],
                                'E1': m_E[4], 'E3': m_E[5]
                            }
                    continue

                if lendo_nos_result:
                    if "LARGEST INDIVIDUAL RESIDUAL" in linha or linha_strip == "": lendo_nos_result = False
                    else:
                        partes = linha.split()
                        if len(partes) >= 7 and partes[0].isdigit():
                            id_no = int(partes[0])
                            self.historico_nos[id_no][passo_global] = {'dX': float(partes[5]), 'dZ': float(partes[6])}
                            
        self.lista_materiais = sorted(list(set(self.materiais.values())))

    def _calcular_escalas_base(self):
        x_coords = [n['X'] for n in self.nos.values()]
        self.largura_barragem = max(x_coords) - min(x_coords) if x_coords else 100
        max_d = max([np.sqrt(p['dX']**2 + p['dZ']**2) for no in self.historico_nos.values() for p in no.values()] + [0])
        self.max_desloc_global = max_d if max_d > 0 else 0.001

    def gerar_triangulacao_ativa(self, passo, materiais_ativos, forcar_tudo=False):
        node_ids = sorted(self.nos.keys())
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        x = [self.nos[nid]['X'] for nid in node_ids]
        z = [self.nos[nid]['Z'] for nid in node_ids]
        
        triangulos, mask = [], []
        nos_ativos = set()
        
        for el, conec in self.elementos.items():
            ativo = True if forcar_tudo else (passo in self.historico_elem[el]) and (self.materiais[el] in materiais_ativos)
            n1, n2, n3, n4 = [node_to_idx[conec[i]] for i in (0, 2, 4, 6)]
            triangulos.extend([[n1, n2, n3], [n1, n3, n4]])
            mask.extend([not ativo, not ativo])
            if ativo: nos_ativos.update([conec[0], conec[2], conec[4], conec[6]])
            
        triang = mtri.Triangulation(x, z, triangulos)
        triang.set_mask(mask)
        return triang, node_to_idx, nos_ativos

    def _interpolar_para_nos(self, passo, variavel):
        valores_nos = np.zeros(len(self.nos))
        contagem = np.zeros(len(self.nos))
        node_to_idx = {nid: i for i, nid in enumerate(sorted(self.nos.keys()))}
        
        for el, conec in self.elementos.items():
            if passo in self.historico_elem[el]:
                val = self.historico_elem[el][passo].get(variavel, 0)
                for no in conec:
                    idx = node_to_idx[no]
                    valores_nos[idx] += val
                    contagem[idx] += 1
                    
        with np.errstate(invalid='ignore'):
            valores_nos = np.divide(valores_nos, contagem)
            valores_nos = np.nan_to_num(valores_nos, nan=0.0, posinf=0.0, neginf=0.0)
        return valores_nos


# ==============================================================================
# 2. INTERFACE WEB (STREAMLIT APP)
# ==============================================================================
st.set_page_config(page_title="PROGEO 2D Viewer", layout="wide")

@st.cache_resource
def carregar_modelo(file_bytes):
    # Salva temporariamente para o Leitor abrir, mas mantém em cache de memória
    with open("temp.pri", "wb") as f:
        f.write(file_bytes)
    return LeitorPROGEO("temp.pri")

st.title("Visualizador PROGEO 2D")
uploaded_file = st.file_uploader("Faça o upload do seu arquivo .PRI", type=["pri", "txt", "out"])

if uploaded_file is not None:
    progeo = carregar_modelo(uploaded_file.getvalue())
    escala_base_visual = (progeo.largura_barragem * 0.05) / progeo.max_desloc_global
    
    # ---- BARRA LATERAL (CONTROLES) ----
    st.sidebar.header("Controles da Malha")
    passo = st.sidebar.slider("Passo Global:", 1, progeo.total_passos, progeo.total_passos)
    variavel = st.sidebar.selectbox("Isolinha de Cor:", ['Geometria Base', 'SZZ', 'SXX', 'S1', 'S3', 'PWP', 'RM', 'EZZ', 'EXX', 'E1', 'E3'])
    mats_ativos = st.sidebar.multiselect("Materiais Ativos:", progeo.lista_materiais, default=progeo.lista_materiais)
    
    st.sidebar.markdown("---")
    ver_def = st.sidebar.checkbox("Rede Deformada")
    mult_def = st.sidebar.number_input("Escala Deformada:", value=1.0, step=1.0)
    ver_vet = st.sidebar.checkbox("Vetores de Deslocamento")
    mult_vet = st.sidebar.number_input("Escala Vetor:", value=1.0, step=0.5)
    
    st.sidebar.markdown("---")
    ver_cruz = st.sidebar.checkbox("Cruzes de Tensão")
    esc_cruz = st.sidebar.number_input("Escala Cruz:", value=0.05, step=0.01)
    ver_id_nos = st.sidebar.checkbox("IDs dos Nós")
    ver_id_el = st.sidebar.checkbox("IDs dos Elementos")
    proporcao_real = st.sidebar.checkbox("Proporção Real 1:1", value=True)

    # ---- ABAS DA INTERFACE ----
    aba_malha, aba_graficos = st.tabs(["Visualização 2D (Malha)", "Gráficos Analíticos"])
    
    # ---------------------------------------------------------
    # ABA 1: MALHA
    # ---------------------------------------------------------
    with aba_malha:
        fig_malha, ax_malha = plt.subplots(figsize=(12, 6))
        
        triang_fundo, _, _ = progeo.gerar_triangulacao_ativa(passo, mats_ativos, forcar_tudo=True)
        ax_malha.triplot(triang_fundo, color='lightgray', linewidth=0.3, alpha=0.4)
        triang, map_nos, nos_ativos = progeo.gerar_triangulacao_ativa(passo, mats_ativos)
        
        if np.all(triang.mask):
            ax_malha.text(0.5, 0.5, "Nenhum elemento ativo.", ha='center', color='red', transform=ax_malha.transAxes)
        else:
            if variavel != 'Geometria Base':
                valores = progeo._interpolar_para_nos(passo, variavel)
                idx_ativos = [map_nos[n] for n in nos_ativos]
                valores_ativos = valores[idx_ativos] if len(idx_ativos) > 0 else [0, 1]
                v_min, v_max = np.min(valores_ativos), np.max(valores_ativos)
                
                if np.isclose(v_min, v_max): v_min, v_max = v_min - 0.1, v_max + 0.1
                niveis = np.linspace(v_min, v_max, 12)
                
                contorno = ax_malha.tricontourf(triang, valores, levels=niveis, cmap='jet', extend='both')
                ax_malha.triplot(triang, color='white', linewidth=0.1, alpha=0.3)
                fig_malha.colorbar(contorno, ax=ax_malha, label=variavel)
            else:
                ax_malha.triplot(triang, color='gray', linewidth=0.8)

        if ver_def and not np.all(triang.mask):
            fator_def = escala_base_visual * mult_def
            x_def = [progeo.nos[n]['X'] + progeo.historico_nos[n].get(passo, {'dX': 0.0})['dX'] * fator_def for n in sorted(progeo.nos.keys())]
            z_def = [progeo.nos[n]['Z'] + progeo.historico_nos[n].get(passo, {'dZ': 0.0})['dZ'] * fator_def for n in sorted(progeo.nos.keys())]
            tri_def = mtri.Triangulation(x_def, z_def, triang.triangles)
            tri_def.set_mask(triang.mask)
            ax_malha.triplot(tri_def, color='black', linewidth=0.8, alpha=0.6)

        if ver_vet and len(nos_ativos) > 0:
            fator_vet = escala_base_visual * mult_vet
            X_vet, Z_vet, U_vet, V_vet = [], [], [], []
            for n in nos_ativos:
                X_vet.append(progeo.nos[n]['X'])
                Z_vet.append(progeo.nos[n]['Z'])
                U_vet.append(progeo.historico_nos[n].get(passo, {'dX': 0.0})['dX'] * fator_vet)
                V_vet.append(progeo.historico_nos[n].get(passo, {'dZ': 0.0})['dZ'] * fator_vet)
            ax_malha.quiver(X_vet, Z_vet, U_vet, V_vet, color='darkmagenta', angles='xy', scale_units='xy', scale=1, width=0.003, zorder=5)

        for el, hist in progeo.historico_elem.items():
            if passo not in hist or progeo.materiais[el] not in mats_ativos: continue
            nos_el = progeo.elementos[el]
            cantos = [nos_el[0], nos_el[2], nos_el[4], nos_el[6]]
            xc, zc = np.mean([progeo.nos[n]['X'] for n in cantos]), np.mean([progeo.nos[n]['Z'] for n in cantos])
            
            if ver_cruz:
                s1, s3 = hist[passo]['S1'] * esc_cruz, hist[passo]['S3'] * esc_cruz
                ang = np.deg2rad(hist[passo].get('ANGLE', 0))
                dx1, dz1 = s1 * np.cos(ang), s1 * np.sin(ang)
                dx3, dz3 = s3 * np.cos(ang + np.pi/2), s3 * np.sin(ang + np.pi/2)
                ax_malha.plot([xc-dx1, xc+dx1], [zc-dz1, zc+dz1], color='red' if hist[passo]['S1'] < 0 else 'blue', linewidth=1.5)
                ax_malha.plot([xc-dx3, xc+dx3], [zc-dz3, zc+dz3], color='red' if hist[passo]['S3'] < 0 else 'blue', linewidth=1.5)
                
            if ver_id_el: ax_malha.text(xc, zc, str(el), fontsize=8, color='maroon', weight='bold', ha='center', va='center', zorder=10)

        if ver_id_nos:
            for n_id, n_data in progeo.nos.items():
                if n_id in nos_ativos: ax_malha.text(n_data['X'], n_data['Z'], str(n_id), fontsize=7, color='black', ha='center', va='center', zorder=10)

        info = progeo.passo_info.get(passo, {'Estagio': '-', 'Inc': '-'})
        titulo = f"Passo Global: {passo} | Estágio: {info['Estagio']} | Incremento: {info['Inc']}"
        ax_malha.set_title(titulo, fontsize=12, weight='bold')

        if proporcao_real: ax_malha.set_aspect('equal')
        else: ax_malha.set_aspect('auto')
            
        ax_malha.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig_malha)

    # ---------------------------------------------------------
    # ABA 2: GRÁFICOS ANALÍTICOS
    # ---------------------------------------------------------
    with aba_graficos:
        col1, col2, col3 = st.columns(3)
        with col1:
            tipo_graf = st.selectbox("Análise:", ['Trajetória p-q (MIT)', 'Trajetória p-q (Cambridge)', 'Carga x Deslocamento (Nó)', 'Valores em Seção de Reta'])
            id_alvo = st.number_input("ID (Nó/Elemento):", value=1)
        with col2:
            ver_env = st.checkbox("Plotar Envoltória de Ruptura", value=True)
            c_linha = st.number_input("Coesão (c'):", value=0.0)
            phi_linha = st.number_input("Ângulo de Atrito (φ'):", value=30.0)
        with col3:
            x0 = st.number_input("X Início (Seção):", value=0.0)
            z0 = st.number_input("Z Início (Seção):", value=0.0)
            x1 = st.number_input("X Fim (Seção):", value=150.0)
            z1 = st.number_input("Z Fim (Seção):", value=0.0)
            
        if st.button("Gerar Gráfico"):
            fig_g, ax_g = plt.subplots(figsize=(10, 5))
            df_export_dict = {}
            
            if 'Trajetória' in tipo_graf:
                if id_alvo in progeo.historico_elem:
                    hist = progeo.historico_elem[id_alvo]
                    passos = sorted(hist.keys())
                    p_vals, q_vals = [], []
                    
                    for p in passos:
                        s1, s3, syy = hist[p]['S1'], hist[p]['S3'], hist[p]['SYY']
                        if 'MIT' in tipo_graf:
                            p_vals.append(-(s1 + s3) / 2.0)
                            q_vals.append(abs(s1 - s3) / 2.0)
                        else:
                            p_vals.append(-(s1 + syy + s3) / 3.0)
                            q_vals.append((1/np.sqrt(2)) * np.sqrt((s1-syy)**2 + (syy-s3)**2 + (s3-s1)**2))
                    
                    ax_g.plot(p_vals, q_vals, '-o', color='purple', label='Caminho de Tensão')
                    df_export_dict = {'Passo_Global': pd.Series(passos), 'P_Efetivo': pd.Series(p_vals), 'Q_Desviador': pd.Series(q_vals)}
                    
                    if ver_env:
                        phi_rad = np.deg2rad(phi_linha)
                        p_line = np.linspace(0, max(p_vals)*1.2 if p_vals else 100, 100)
                        if 'MIT' in tipo_graf:
                            q_line = p_line * np.sin(phi_rad) + c_linha * np.cos(phi_rad)
                            ax_g.plot(p_line, q_line, 'r--', label=f"Kf-line (φ'={phi_linha}°, c'={c_linha})")
                        else:
                            M = (6 * np.sin(phi_rad)) / (3 - np.sin(phi_rad))
                            q_line = M * p_line + c_linha * (6 * np.cos(phi_rad)) / (3 - np.sin(phi_rad))
                            ax_g.plot(p_line, q_line, 'r--', label=f"M-line (φ'={phi_linha}°, c'={c_linha})")
                        
                        df_export_dict['P_Envoltoria'] = pd.Series(p_line)
                        df_export_dict['Q_Envoltoria'] = pd.Series(q_line)
                            
                    ax_g.set_title(f'{tipo_graf} - Elemento {id_alvo}')
                    ax_g.set_xlabel("Tensão Efetiva Média (p')")
                    ax_g.set_ylabel("Tensão Desviadora (q)")
                    ax_g.legend()
                    
            elif 'Carga' in tipo_graf:
                if id_alvo in progeo.historico_nos:
                    hist = progeo.historico_nos[id_alvo]
                    passos = sorted(hist.keys())
                    dz_vals = [hist[p]['dZ'] for p in passos]
                    ax_g.plot(passos, dz_vals, '-s', color='darkblue')
                    ax_g.set_title(f'Deslocamento Vertical (Nó {id_alvo})')
                    ax_g.set_xlabel("Passo Global (Acumulado)")
                    ax_g.set_ylabel("Recalque (m)")
                    df_export_dict = {'Passo_Global': pd.Series(passos), 'Recalque_dZ': pd.Series(dz_vals)}
                    
            elif 'Seção' in tipo_graf:
                var_sec = variavel if variavel != 'Geometria Base' else 'PWP'
                x_lin = np.linspace(x0, x1, 50)
                z_lin = np.linspace(z0, z1, 50)
                triang, _, _ = progeo.gerar_triangulacao_ativa(passo, progeo.lista_materiais)
                valores_nos = progeo._interpolar_para_nos(passo, var_sec)
                try:
                    interpolador = mtri.LinearTriInterpolator(triang, valores_nos)
                    v_lin = interpolador(x_lin, z_lin)
                    dist = np.sqrt((x_lin - x0)**2 + (z_lin - z0)**2)
                    ax_g.plot(dist, v_lin, color='teal', linewidth=2)
                    ax_g.set_title(f'{var_sec} ao longo da linha no Passo {passo}')
                    ax_g.set_xlabel('Distância (m)')
                    df_export_dict = {'Distancia_m': pd.Series(dist), 'X_coord': pd.Series(x_lin), 'Z_coord': pd.Series(z_lin), 'Valor': pd.Series(v_lin)}
                except Exception: pass

            ax_g.grid(True, linestyle='--')
            st.pyplot(fig_g)
            
            # Botão de Download Nativo
            df = pd.DataFrame(df_export_dict)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar Dados (CSV)",
                data=csv,
                file_name='dados_analiticos_progeo.csv',
                mime='text/csv'
            )
