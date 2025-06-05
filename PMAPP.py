import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from datetime import date
import io

#%% Units
inch = 1
sqin = 1
ft = 12 * inch
lb = 1
kip = 1000 * lb
psi = lb / sqin
ksi = 1000 * psi

#%% Streamlit Setup
st.set_page_config(page_title="P-M Curve Generator", layout="centered")
st.title("P-M Curve Generator for Circular Concrete Columns and Shaft")
st.sidebar.header("Design Parameters")
#%% Sidebar Input

project_name = st.sidebar.text_input("Project Name", value="")
designer_name = st.sidebar.text_input("Designer Name", value="")
diameter = st.sidebar.number_input("Column Diameter (in)", value=36.0, step=1.0) * inch
cover = st.sidebar.number_input("Concrete Cover (in)", value=4.0, step=1.0) * inch
number_of_bars = st.sidebar.number_input("Number of Bars", value=12, step=1)
bar = int(st.sidebar.selectbox("Bar Size", [f"#{i}" for i in range(3, 12)]).replace("#", ""))
fc = st.sidebar.number_input("Concrete Strength f'c (ksi)", value=4.0, step=1.0) * ksi
fy = st.sidebar.number_input("Steel Yield Strength fy (ksi)", value=60.0, step=1.0) * ksi
tie_type = st.sidebar.radio("Transverse Reinforcement Type", ["Spirals", "Hoops"])
export_pdf = st.sidebar.checkbox("Export PDF Report")
trans_bar_coeff = 0.85 if tie_type == "Spirals" else 0.80

radius = diameter / 2

#%% Steel Lookup
steel_lookup_data = {
    "Bar": [3, 4, 5, 6, 7, 8, 9, 10, 11],
    "Area": [0.11, 0.20, 0.31, 0.44, 0.60, 0.79, 1.00, 1.27, 1.56],
    "Dia": [0.375, 0.500, 0.625, 0.750, 0.875, 1.000, 1.128, 1.270, 1.410]
}
steel_table = pd.DataFrame(steel_lookup_data)
bar_row = steel_table[steel_table['Bar'] == bar]
if not bar_row.empty:
    area_bar = bar_row.iloc[0]["Area"]
    d_bar = bar_row.iloc[0]["Dia"]
else:
    st.stop()

Es = 29000 * ksi
ys = fy / Es
crush_strain = -0.003
gamma_conc = 0.135
Ec = 120000 * ((gamma_conc) ** 2) * (fc / ksi) ** 0.33

#%% Max Pn

area_shaft = diameter**2*np.pi/4
area_steel = area_bar*number_of_bars
area_conc_total = area_shaft-area_steel

Max_Pn = 0.75*trans_bar_coeff*(area_conc_total*0.85*fc+area_steel*fy)

#%% Functions
def angle(c):
    if c > diameter / 0.85:
        return np.pi
    else:
        return np.arccos((radius - c * 0.85) / radius)

def circle_sector(c, theta):
    if theta < 2 * np.pi:
        area = ((radius ** 2) / 2) * (theta * 2 - np.sin(2 * theta))
    else:
        area = diameter ** 2 * np.pi / 4
    cg = (4 * radius * np.sin(theta) ** 3) / (3 * (theta * 2 - np.sin(theta * 2)))
    return area, cg

def resist_factor(strain):
    if strain <= 0.002:
        return 0.75
    elif strain < 0.005:
        return 0.75 + 0.15 * (strain - 0.002) / (0.005 - 0.002)
    else:
        return 0.9

def steel_stress(strain):
    if abs(strain) < ys:
        stress = abs(strain) * Es
    else:
        stress = fy
    return stress if strain >= 0 else -stress

#%% Rebar Layout
angle_space = 360 / number_of_bars
steel_cg = radius - cover - d_bar / 2
bar_coords = {
    barid: (steel_cg * np.cos(np.deg2rad(barid * angle_space)),
            steel_cg * np.sin(np.deg2rad(barid * angle_space)))
    for barid in range(number_of_bars)
}

fig1, ax = plt.subplots(figsize=(8.5, 11))
ax.set_aspect('equal')
circle = plt.Circle((0, 0), radius, color='gray', fill=False, linewidth=2)
ax.add_artist(circle)
for i, (x, y) in bar_coords.items():
    ax.plot(x, y, 'o', color='purple')
    ax.text(x, y + 0.5, str(i + 1), color='black', fontsize=10, ha='center')
ax.axhline(0, color='black', linewidth=0.8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlim(-radius - 2, radius + 2)
ax.set_ylim(-radius - 2, radius + 2)
ax.set_title('Rebar Layout in Circular Shaft')
ax.set_xlabel('X (in)')
ax.set_ylabel('Y (in)')
ax.grid(True)
plt.tight_layout()
st.pyplot(fig1)

#%% P-M Curve Calculation
c_fine = np.arange(2, 100, 0.1)
P_M_Curve = []
for c in c_fine:
    bar_cg_force_moment = []
    Asminus = 0
    for barid, (x, y) in bar_coords.items():
        dist_geo_cg = y
        y_bar_top = radius - dist_geo_cg
        if y_bar_top <= c:
            bar_strain = crush_strain * (c - y_bar_top) / c
        else:
            bar_strain = -crush_strain * (y_bar_top - c) / c
        bar_stress = steel_stress(bar_strain)
        bar_force = bar_stress * area_bar
        bar_moment = bar_force * dist_geo_cg
        bar_cg_force_moment.append((bar_strain, bar_force, bar_moment))
        if y_bar_top < 0.85*c:
            Asminus += area_bar
    bar_cg_force_moment = np.array(bar_cg_force_moment)
    theta = angle(c)
    compression_area, concrete_cg = circle_sector(c, theta)
    concrete_force = -1 * (compression_area - Asminus) * 0.85 * fc
    concrete_moment = concrete_force * concrete_cg
    Pn = concrete_force + sum(bar_cg_force_moment[:, 1])
    Mn = concrete_moment + sum(bar_cg_force_moment[:, 2])
    phi = resist_factor(max(bar_cg_force_moment[:, 0]))
    Pr = Pn * phi
    if abs(Pr) > Max_Pn:
        Pr = -Max_Pn
    Mr = Mn * phi
    P_M_Curve.append((Pn, Mn, Pr, Mr))

P_M_Curve = np.array(P_M_Curve)
Pn_vals = -P_M_Curve[:, 0] / kip
Mn_vals = -P_M_Curve[:, 1] / (kip * ft)
Pr_vals = -P_M_Curve[:, 2] / kip
Mr_vals = -P_M_Curve[:, 3] / (kip * ft)

fig2 = plt.figure(figsize=(8.5, 11))
plt.plot(Mn_vals, Pn_vals, label='Nominal (Pn-Mn)', color='blue', linewidth=2)
plt.plot(Mr_vals, Pr_vals, label='Factored (Pr-Mr)', color='red', linestyle='--', linewidth=2)
plt.title('P-M Interaction Curve for Circular Concrete Column')
plt.xlabel('Moment (kip-ft)')
plt.ylabel('Axial Load (kip)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.axhline(y=0, color='black', linewidth=1)
plt.axvline(x=0, color='black', linewidth=1)
plt.legend()
# Add major and minor grid
plt.grid(True, which='major', linestyle='--', linewidth=0.75)
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth=0.5)

# Ticks every 250 units
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(250),)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(250))

# Add zero lines for x=0 and y=0
plt.axhline(y=0, color='black', linewidth=1.5)
plt.axvline(x=0, color='black', linewidth=0.2)

# Legend, formatting
plt.legend()
plt.tick_params(which='major', length=6, width=1)
plt.tick_params(which='minor', length=3, width=0.5)

plt.axis('tight')
# Add tick limiter to avoid warning
ax = plt.gca()
plt.tight_layout()
st.pyplot(fig2)
#%% Assumptions
with st.expander("Assumptions"):
    st.markdown("""
      ASSUMPTIONS & TECHNICAL GUIDANCE
      
      General:
      - Concrete crushing strain assumed = 0.003
      - Plane sections remain plane (linear strain profile)
      - Stress in steel follows ideal elastic-plastic behavior
      - Concrete stress follows Whitney Stress Block Method (0.85 * f'c)
      - Area of bars are removed from concrete compression zones by 
        their full area as soon as the CG of the bar is the compression zone.
      
      Bar Reinforcement:
      - ASTM standard sizes and areas used
      - Bar CG locations are radial and uniformly distributed
      - Yield strain (ey) = fy / Es
      
      Design Standard:
      - AASHTO LRFD 9th Edition
      - Strength reduction φ varies from 0.75 to 0.9 based on strain
      
      Limitations:
      - Only valid for circular sections
      - No slenderness, buckling, or second-order effects
      - Outside of pure compression the effects of confinement 
       or lack thereof are ignored
    """, unsafe_allow_html=True)
with st.expander("Technical Details"):
    st.markdown("""
  Verison: 1.0
  
  Steps this analysis takes to create the curves:
  1. Assume a neutral axis depth
  2. Assume concrete is crushing at most extreme compression
     fiber
  3. Use Whitney Stress Block to find area in compression
  4. Solve for the force and moment produced by compression
  5. Solve for where bar CGs are in relation to neutral axis
  6. Solve for strains of individual bars
  7. Solve for bar forces and moments
  8. Solve for resistance factor using most extreme tensile rebar
     strain
  9. Sum forces to find Pn
  10. Sum moments for Mn
  10. Factor Pn and Mn to solve for Pr and Mr
  11. Impose restriction on Pn so that it cannot be greater 
      than maximum axial load, including confiment effects

  Technical Details
  - Bending is about the x-axis
  - Neutral axis depth is measured from the most extreme compression 
    fiber
  - The strain profile is assumed perfectly linear
  - Strains are calculated using similar triangles
  - The range of assumed neutral axis depths (c) start at 2in and 
    go to 100in, with 0.1in steps. Anything lower than 2in for NA is too 
    tensile. At these point I do not need to know about tensile interaction
    but later this may be updated. 
  - Full areas of steel are subtracted from the compression zone when their
    CG falls within 0.85*c. This will lead to slightly lower compressive
    forces than in reality until the 0.85*c is greater than the CG depth of 
    the bar + 0.5*diameter_bar. Really it saves a lot of work to do it this way
  - Steel material model is perfectly elastic-plastic
  - Concrete material model is Whitney Stress Block
  
""", unsafe_allow_html=True)
with st.expander("Credits"):
    st.markdown("""
ITH 
Full code is free to use and expand: https://github.com/OpenSourcePT/PMCurve
""", unsafe_allow_html=True)
#%% PDF Export
if export_pdf:
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        # Title Page
        fig_title, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        today = date.today().strftime("%B %d, %Y")
        ax.text(0.5, 0.75, "P-M Interaction Curve", fontsize=24, ha="center")
        ax.text(0.5, 0.65, "Circular Concrete Drilled Shaft or Column", fontsize=18, ha="center")
        ax.text(0.5, 0.55, "AASHTO LRFD 9th Edition-Based Design  ", fontsize=14, ha="center")
        if project_name:
            ax.text(0.5, 0.45, f"Project: {project_name}", fontsize=12, ha="center")
        ax.text(0.5, 0.35, f"Designer: {designer_name}", fontsize=12, ha="center")
        ax.text(0.5, 0.25, f"Date: {today}", fontsize=12, ha="center")
        pdf.savefig(fig_title)
        plt.close(fig_title)

        # Parameters Page
        fig_params, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        bar_coord_list = "\n    ".join([f"Bar {i+1:2d}: x = {x:.2f} in, y = {y:.2f} in" for i, (x, y) in bar_coords.items()])
        param_text = f"""
Design Parameters Summary

• Column Diameter         : {diameter:.1f} in
• Concrete Cover          : {cover:.1f} in
• Number of Bars          : {number_of_bars}
• Bar Size (ASTM)         : #{bar}
• Bar Diameter            : {d_bar:.3f} in
• Bar Area                : {area_bar:.3f} in²
• Total Steel Area (As)   : {area_bar * number_of_bars:.3f} in²

Material Properties

• Steel Yield Strength (fy)   : {fy / ksi:.1f} ksi
• Steel Youngs Modulus (Es)   : {Es / ksi:.1f} ksi
• Steel Yield Strain (εy)     : {ys:.6f}
• Concrete Strength (f'c)     : {fc / ksi:.1f} ksi
• Concrete Crushing Strain    : 0.003

Steel Bar Centroids (in):
    {bar_coord_list}
"""
        ax.text(0.01, 0.95, param_text, fontsize=11, va='top', ha='left', family='monospace')
        pdf.savefig(fig_params)
        plt.close(fig_params)

        # Assumptions & Guidance Page
        fig_notes, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        notes_text = """
ASSUMPTIONS

General:
- Concrete crushing strain assumed = 0.003
- Plane sections remain plane (linear strain profile)
- Stress in steel follows ideal elastic-plastic behavior
- Concrete stress follows Whitney Stress Block Method (0.85 * f'c)
- Area of bars are removed from concrete compression zones by 
  their full area as soon as the CG of the bar is the compression zone.

Bar Reinforcement:
- ASTM standard sizes and areas used
- Bar CG locations are radial and uniformly distributed
- Yield strain (εy) = fy / Es

Design Standard:
- AASHTO LRFD 9th Edition
- Strength reduction φ varies from 0.75 to 0.9 based on strain

Limitations:
- Only valid for circular sections
- No slenderness, buckling, or second-order effects
- Outside of pure compression the effects of confinement 
  or lack thereof are ignored
"""
        ax.text(0.01, 0.95, notes_text, fontsize=11, va='top', ha='left', family='monospace')
        pdf.savefig(fig_notes)
        plt.close(fig_notes)

        # Add plots
        pdf.savefig(fig1)
        pdf.savefig(fig2)

    # Move this directly after the export_pdf checkbox section
    today_str = date.today().strftime("%Y-%m-%d")
    safe_project_name = project_name.strip().replace(" ", "_") or "UnnamedProject"
    pdf_filename = f"{safe_project_name}_PM_Curve_{today_str}.pdf"

    st.sidebar.download_button(
    label="📥 Download PDF Report",
    data=pdf_buffer.getvalue(),
    file_name=pdf_filename,
    mime="application/pdf"
)
