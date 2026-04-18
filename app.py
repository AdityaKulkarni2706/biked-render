import streamlit as st
import pandas as pd
import bike_pipeline_claude as bp
import os

@st.cache_resource
def load_pipeline():
    template_path = r"Biked_Reference_Data\PlainRoadbikestandardized.txt"
    bcad_dir = r"Biked_Reference_Data\output\bcad"
    pipeline = bp.BikePipeline(template_path, bcad_dir)
    df = pd.read_csv(r"Biked_Reference_Data\clip_sBIKED_processed.csv") 
    base_row = df.iloc[0].copy()
    return pipeline, base_row, bcad_dir

pipeline, base_row, bcad_dir = load_pipeline()


def get_svg_content(filename):
    filepath = os.path.join(bcad_dir, f"{filename}.svg")
    with open(filepath, "r") as file:
        return file.read()

if "current_bike_image" not in st.session_state:
    with st.spinner("Rendering initial bike..."):
        # Generate the base bike immediately 
        initial_filename = "initial_base_bike"
        pipeline.row_to_svg(base_row, initial_filename)
        
        # Save the SVG string into Streamlit's memory
        st.session_state.current_bike_image = get_svg_content(initial_filename)

st.title("Interactive Bike Modifier")
with st.sidebar:
    head_angle = st.slider("Head Angle (°)", min_value=60.0, max_value=90.0, value=float(base_row['Head angle']))
    seat_angle = st.slider("Seat Angle (°)", min_value=60.0, max_value=90.0, value=float(base_row['Seat angle']))
    seat_tube_length = st.slider("Seat Tube Length (mm)", min_value=300.0, max_value=700.0, value=float(base_row['Seat tube length']))
    stack = st.slider("Stack (mm)", min_value=400.0, max_value=700.0, value=float(base_row['Stack']))
    chainstay = st.slider("Chainstay Length (mm)", min_value=350.0, max_value=600.0, value=float(base_row['CS textfield']))
    bb_drop = st.slider("BB Drop/Height (mm)", min_value=0.0, max_value=150.0, value=float(base_row['BB textfield']))
    ht_length = st.slider("Head Tube Length (mm)", min_value=80.0, max_value=300.0, value=float(base_row['Head tube length textfield']))
    
    st.header("Fit & Tubing")
    saddle_height = st.slider("Saddle Height (mm)", min_value=500.0, max_value=950.0, value=float(base_row['Saddle height']))
    ttd = st.slider("Top Tube Diameter (mm)", min_value=20.0, max_value=60.0, value=float(base_row['ttd']))
    dtd = st.slider("Down Tube Diameter (mm)", min_value=20.0, max_value=80.0, value=float(base_row['dtd']))

    st.divider()
    render_button = st.button("Render Bike", type="primary", use_container_width=True)


if render_button:
    with st.spinner("Pipeline is generating the bike..."):
        current_row = base_row.copy()
        current_row['Head angle'] = head_angle
        current_row['Seat angle'] = seat_angle
        current_row['Seat tube length'] = seat_tube_length
        current_row['Stack'] = stack
        current_row['CS textfield'] = chainstay
        current_row['BB textfield'] = bb_drop
        current_row['Head tube length textfield'] = ht_length
        current_row['Saddle height'] = saddle_height
        current_row['ttd'] = ttd
        current_row['dtd'] = dtd
        filename = "streamlit_demo_render"
        pipeline.row_to_svg(current_row, filename)
        st.session_state.current_bike_image = get_svg_content(filename)

if "current_bike_image" in st.session_state:
    st.image(st.session_state.current_bike_image)

        