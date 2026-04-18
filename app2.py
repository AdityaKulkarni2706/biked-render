import streamlit as st
import pandas as pd
from PIL import Image
from bike_engine import BikeSearchEngine
from bike_pipeline_claude import BikePipeline  # Import your new pipeline

st.set_page_config(page_title="BIKED++ Explorer", layout="wide")
st.title("🚲 BIKED++ Multimodal Search Engine")

# --- 1. Safe Data Loading ---
@st.cache_resource(show_spinner="Booting AI Model & loading 1.4M bikes...")
def load_system():
    # 1. Load the search engine
    emb_path = r"C:\Users\Adi\Downloads\all_embeddings.npy"
    prm_path = r"C:\Users\Adi\Downloads\all_parametric.npy"
    engine = BikeSearchEngine(emb_path, prm_path)
    
    # 2. Load the column names from your CSV so the XML injector knows what's what
    # We only need the headers, so nrows=0 is a fast way to grab them
    csv_path = r"Biked_Reference_Data\clip_sBIKED_processed.csv"
    df_headers = pd.read_csv(csv_path, nrows=0)
    
    # Exclude non-parametric columns if necessary (like 'Unnamed: 0', 'Image_Path', etc.)
    # Adjust this slicing based on exactly which 96 columns match your .npy file
    col_names = df_headers.columns[-96:].tolist() 
    
    return engine, col_names

engine, col_names = load_system()

# --- 2. The Integration Function ---
def render_my_bike(parameters, bike_index, cols):
    """Converts the raw numpy array to a Series, runs the pipeline, and displays the SVG."""
    
    # 1. Create the Pandas Series your pipeline expects
    bike_series = pd.Series(parameters, index=cols)
    
    # 2. Run your pipeline using the context manager
    with st.spinner("Rendering CAD to SVG..."):
        with BikePipeline() as pipeline:
            svg_path = pipeline.row_to_svg(bike_series, "bike_test")
            
    # 3. Display the SVG in Streamlit
    # Streamlit natively renders SVGs using st.image if you pass the file path
    st.image(str(svg_path), use_column_width=True)
    st.success(f"Successfully rendered: {svg_path.name}")

# --- 3. The UI ---
tab_text, tab_image = st.tabs(["💬 Search by Text", "🖼️ Search by Image"])

with tab_text:
    text_query = st.text_input("Example: 'A rugged mountain bike with thick tires'")
    if st.button("Search by Text", type="primary"):
        if text_query:
            results = engine.search_by_text(text_query, top_k=3)
            cols_ui = st.columns(3)
            for i, res in enumerate(results):
                with cols_ui[i]:
                    st.metric(label=f"Match #{i+1}", value=f"{res['score']:.3f} Score")
                    # Pass the params, the index (for the filename), and the column names
                    render_my_bike(res['parameters'], res['index'], col_names)

with tab_image:
    uploaded_file = st.file_uploader("Upload a JPG or PNG", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        ref_image = Image.open(uploaded_file)
        st.image(ref_image, caption="Reference", width=300)
        
        if st.button("Search by Image", type="primary"):
            results = engine.search_by_image(ref_image, top_k=3)
            cols_ui = st.columns(3)
            for i, res in enumerate(results):
                with cols_ui[i]:
                    st.metric(label=f"Match #{i+1}", value=f"{res['score']:.3f} Score")
                    render_my_bike(res['parameters'], res['index'], col_names)