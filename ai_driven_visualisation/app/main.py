import streamlit as st
import pandas as pd
import os
import json
from src.data_ingestion.file_handler import handle_file_upload
from src.preprocessing.data_cleaner import clean_data
from src.rag.retriever import retrieve_knowledge
from src.ai_engine.llm_manager import generate_visualization_code
from src.visualization.chart_generator import render_visualization
from src.ai_engine.narrative_generator import generate_insights
from src.orchestration.workflow_manager import trigger_workflow
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# App configuration
st.set_page_config(
    page_title="AI-Driven Visualization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("AI-Driven Visualization")
st.markdown("""
    Upload your data and get intelligent visualizations with narrative insights.
    This tool combines RAG, multimodal processing, and AI to understand your data context.
""")

# Sidebar for file upload and options
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your data file", 
        type=["csv", "xlsx", "json", "pdf", "png", "jpg"]
    )
    
    if uploaded_file:
        file_details = {
            "Filename": uploaded_file.name,
            "FileType": uploaded_file.type,
            "FileSize": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.write("File Details:", file_details)
    
    st.header("Visualization Options")
    viz_type = st.selectbox(
        "Visualization Type",
        ["Auto (AI Selected)", "Bar Chart", "Line Chart", "Scatter Plot", "Heatmap", "Pie Chart"]
    )
    
    narrative_detail = st.slider(
        "Narrative Detail Level", 
        min_value=1, 
        max_value=5, 
        value=3,
        help="Higher values provide more detailed explanations"
    )
    
    st.header("Advanced Options")
    use_rag = st.checkbox("Use RAG for Enhanced Context", value=True)
    show_code = st.checkbox("Show Generated Code", value=False)
    
    process_btn = st.button("Generate Visualization", type="primary")

# Main content area
if uploaded_file is not None:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("Data Preview")
        
        # Handle different file types
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head(), use_container_width=True)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)
                st.dataframe(df.head(), use_container_width=True)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
                st.dataframe(df.head(), use_container_width=True)
            elif uploaded_file.name.endswith('.pdf'):
                st.write("PDF preview not available. Processing content...")
                # PDF processing would happen here
            elif uploaded_file.name.endswith(('.png', '.jpg', '.jpeg')):
                st.image(uploaded_file, caption="Uploaded Image")
                st.write("Image data extraction in progress...")
                # Image processing would happen here
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    with col2:
        st.header("Data Summary")
        if 'df' in locals():
            st.write(f"Rows: {df.shape[0]}")
            st.write(f"Columns: {df.shape[1]}")
            
            # Display basic statistics
            st.subheader("Numerical Statistics")
            if df.select_dtypes(include=['number']).shape[1] > 0:
                st.dataframe(df.describe(), use_container_width=True)
            else:
                st.write("No numerical columns found")
                
            # Display categorical data info
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                st.subheader("Categorical Columns")
                for col in categorical_cols[:3]:  # Limit to first 3 columns
                    st.write(f"{col}: {df[col].nunique()} unique values")
                    
                if len(categorical_cols) > 3:
                    st.write(f"... and {len(categorical_cols) - 3} more categorical columns")

# Processing logic when button is clicked
if uploaded_file is not None and process_btn:
    with st.spinner("Processing your data..."):
        # Create placeholder for results
        viz_container = st.container()
        narrative_container = st.container()
        code_container = st.container()
        
        try:
            # Simulating the processing pipeline
            # In a real implementation, these would call the actual functions
            
            # 1. Save uploaded file temporarily
            file_path = handle_file_upload(uploaded_file)
            
            # 2. Trigger n8n workflow
            workflow_id = trigger_workflow("processing_pipeline", {
                "file_path": file_path,
                "visualization_type": viz_type,
                "use_rag": use_rag
            })
            
            # 3. For demo purposes, let's simulate results
            # In a real app, you'd wait for the n8n workflow to complete
            
            with viz_container:
                st.header("Visualization Results")
                
                # Sample visualization (would be generated by the AI)
                if 'df' in locals():
                    if viz_type == "Auto (AI Selected)" or viz_type == "Bar Chart":
                        if 'df' in locals() and len(df.columns) >= 2:
                            x_col = df.columns[0]
                            y_col = df.select_dtypes(include=['number']).columns[0] if len(df.select_dtypes(include=['number']).columns) > 0 else df.columns[1]
                            
                            fig = go.Figure(data=[go.Bar(x=df[x_col].head(10), y=df[y_col].head(10))])
                            fig.update_layout(title=f"{y_col} by {x_col}", xaxis_title=x_col, yaxis_title=y_col)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Line Chart":
                        if 'df' in locals() and len(df.columns) >= 2:
                            x_col = df.columns[0]
                            y_col = df.select_dtypes(include=['number']).columns[0] if len(df.select_dtypes(include=['number']).columns) > 0 else df.columns[1]
                            
                            fig = go.Figure(data=[go.Scatter(x=df[x_col].head(10), y=df[y_col].head(10), mode='lines+markers')])
                            fig.update_layout(title=f"{y_col} Trend by {x_col}", xaxis_title=x_col, yaxis_title=y_col)
                            st.plotly_chart(fig, use_container_width=True)
            
            with narrative_container:
                st.header("AI-Generated Insights")
                
                # Sample narrative (would be generated by the AI)
                st.markdown("""
                    ### Key Findings
                    
                    Based on the analysis of your data, here are the main insights:
                    
                    1. There appears to be a significant correlation between variables X and Y
                    2. The data shows a clear upward trend over the analyzed period
                    3. Outliers were detected in approximately 5% of the data points
                    
                    ### Recommendations
                    
                    - Consider further investigation of the relationship between X and Y
                    - The seasonal patterns suggest timing your strategy accordingly
                    - Data quality could be improved by addressing missing values
                """)
                
            if show_code:
                with code_container:
                    st.header("Generated Visualization Code")
                    
                    # Sample code (would be generated by the AI)
                    code = """
                    import plotly.graph_objects as go
                    import pandas as pd
                    
                    # Load and prepare data
                    df = pd.read_csv('your_data.csv')
                    
                    # Create visualization
                    fig = go.Figure(data=[
                        go.Bar(x=df['category'], y=df['value'])
                    ])
                    
                    # Customize layout
                    fig.update_layout(
                        title='Category Distribution',
                        xaxis_title='Category',
                        yaxis_title='Value',
                        template='plotly_white'
                    )
                    
                    # Display the figure
                    fig.show()
                    """
                    
                    st.code(code, language="python")
            
            # Add download button for the report
            st.download_button(
                label="Download Full Report (PDF)",
                data=b"Sample PDF content",  # Would be actual PDF in implementation
                file_name="ai_visualization_report.pdf",
                mime="application/pdf",
            )
            
        except Exception as e:
            st.error(f"Error processing data: {e}")

# Footer
st.markdown("---")
st.markdown("AI-Driven Visualization | Powered by RAG, Multi-Modal Processing & n8n")