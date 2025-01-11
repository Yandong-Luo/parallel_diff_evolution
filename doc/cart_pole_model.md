![cart pole system](../img/cart pole system.png)

Set the state variable as $x_1 = pos, x_2 = \theta, x_3 = v, x_4 = \dot \theta$. The system can be described as:
$$
\dot{\mathbf{x}} = Ax+Bu+E\lambda\\

\dot{\mathbf{x}} = 
\begin{bmatrix}
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
0 & \frac{g m_p}{m_c} & 0 & 0 \\
0 & \frac{g (m_c + m_p)}{l m_c} & 0 & 0
\end{bmatrix}
\mathbf{x}
+
\begin{bmatrix}
0 \\
0 \\
\frac{1}{m_c} \\
\frac{1}{l m_c}
\end{bmatrix}
u
+
\begin{bmatrix}
0 & 0 \\
0 & 0 \\
0 & 0 \\
\frac{1}{l m_p} & -\frac{1}{l m_p}
\end{bmatrix}
\begin{bmatrix}
\lambda_1 \\
\lambda_2
\end{bmatrix}.

$$

$$
E = I + \Delta T \begin{bmatrix}
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
0 & \frac{g m_p}{m_c} & 0 & 0 \\
0 & \frac{g (m_c + m_p)}{\ell m_c} & 0 & 0
\end{bmatrix}
$$

$$
F = \Delta T \begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
\frac{1}{m_c} & 0 & 0 \\
\frac{1}{\ell m_c} & \frac{1}{\ell m_p} & -\frac{1}{\ell m_p}
\end{bmatrix}
$$

$$
G = \mathbf{0}_{\text{len}_x \times \text{len}_z}
$$

$$
H_1 = \begin{bmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
-1 & \ell & 0 & 0 \\
1 & -\ell & 0 & 0 \\
1 & -\ell & 0 & 0 \\
-1 & \ell & 0 & 0 \\
1 & 0 & 0 & 0 \\
-1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & -1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & -1 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & -1 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{bmatrix},
H_2 = \begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 1 \\
0 & \frac{1}{k_1} & 0 \\
0 & -\frac{1}{k_1} & 0 \\
0 & 0 & \frac{1}{k_2} \\
0 & 0 & -\frac{1}{k_2} \\
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
1 & 0 & 0 \\
-1 & 0 & 0 \\
0 & 1 & 0 \\
0 & -1 & 0 \\
0 & 0 & 1 \\
0 & 0 & -1
\end{bmatrix},
H_3 = \begin{bmatrix}
-\lambda_{\text{max}} & 0 \\
0 & -\lambda_{\text{max}} \\
D_{\text{max}} & 0 \\
0 & 0 \\
0 & D_{\text{max}} \\
0 & 0 \\
0 & 0 \\
0 & 0 \\
0 & 0 \\
0 & 0 \\
0 & 0 \\
0 & 0 \\
0 & 0 \\
0 & 0 \\
0 & 0 \\
0 & 0 \\
0 & 0 \\
0 & 0 \\
0 & 0 \\
0 & 0
\end{bmatrix}
$$

$$
h_\theta = \begin{bmatrix}
0 \\
0 \\
-d_{\text{right}} + D_{\text{max}} \\
d_{\text{right}} \\
-d_{\text{left}} + D_{\text{max}} \\
d_{\text{left}} \\
x_{\text{ub}}[0] \\
-x_{\text{lb}}[0] \\
x_{\text{ub}}[1] \\
-x_{\text{lb}}[1] \\
x_{\text{ub}}[2] \\
-x_{\text{lb}}[2] \\
x_{\text{ub}}[3] \\
-x_{\text{lb}}[3] \\
u_{\text{max}} \\
u_{\text{max}} \\
\lambda_{\text{max}} \\
0 \\
\lambda_{\text{max}} \\
0
\end{bmatrix}
$$

