import streamlit as st
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import io

st.title('CFD Simulation Results Dashboard')

uploaded_file = st.file_uploader('Upload simulation results (.npz or .csv)', type=['npz', 'csv'])

if uploaded_file:
    if uploaded_file.name.endswith('.npz'):
        data = np.load(uploaded_file)
        x = data['x']
        y = data['y']
        mach = data['u'] / np.sqrt(1.4 * 287.0 * data['T'])
        p = data['p']
        T = data['T']
    elif uploaded_file.name.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        x = df['x'].values
        y = df['y'].values
        mach = df['u'].values / np.sqrt(1.4 * 287.0 * df['T'].values)
        p = df['p'].values
        T = df['T'].values
    else:
        st.error('Unsupported file format.')
        st.stop()

    st.sidebar.header('Visualization Controls')
    x_idx = st.sidebar.slider('X index', 0, len(x)-1, 0)
    y_idx = st.sidebar.slider('Y index', 0, len(y)-1, 0)

    st.subheader('2D Contour Plots')
    st.write('Mach Number')
    fig1, ax1 = plt.subplots()
    c1 = ax1.contourf(x, y, mach.T, 50)
    fig1.colorbar(c1, ax=ax1)
    st.pyplot(fig1)
    st.write('Pressure')
    fig2, ax2 = plt.subplots()
    c2 = ax2.contourf(x, y, p.T, 50)
    fig2.colorbar(c2, ax=ax2)
    st.pyplot(fig2)
    st.write('Temperature')
    fig3, ax3 = plt.subplots()
    c3 = ax3.contourf(x, y, T.T, 50)
    fig3.colorbar(c3, ax=ax3)
    st.pyplot(fig3)

    st.subheader('Mach Number Profile')
    st.line_chart(mach[:, y_idx])
    st.subheader('Pressure Profile')
    st.line_chart(p[:, y_idx])
    st.subheader('Temperature Profile')
    st.line_chart(T[:, y_idx])

    st.subheader('Animated Mach Number Evolution')
    if mach.ndim == 2:
        # Fake time evolution for demo: animate over x
        frames = []
        for i in range(mach.shape[0]):
            frames.append(go.Frame(data=[go.Scatter(y=mach[i, :], mode='lines')], name=str(i)))
        fig = go.Figure(
            data=[go.Scatter(y=mach[0, :], mode='lines')],
            layout=go.Layout(
                updatemenus=[dict(type='buttons', showactive=False,
                                  buttons=[dict(label='Play', method='animate', args=[None])])]
            ),
            frames=frames
        )
        st.plotly_chart(fig)
    else:
        st.info('Mach number animation requires 2D data.')

    # Download processed results
    st.subheader('Download Processed Results')
    df_out = pd.DataFrame({'x': np.tile(x, len(y)), 'y': np.repeat(y, len(x)), 'mach': mach.flatten(), 'p': p.flatten(), 'T': T.flatten()})
    csv = df_out.to_csv(index=False).encode('utf-8')
    st.download_button('Download CSV', csv, 'processed_results.csv', 'text/csv') 