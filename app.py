import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import ollama
import os
from typing import Dict, List, Tuple
import re
import json
import logging
import traceback
import sys
import hashlib

# Configure comprehensive logging
def setup_logging():
    """Setup comprehensive logging for the application"""
    # Create logs directory if it doesn't exist
    try:
        if not os.path.exists('logs'):
            os.makedirs('logs')
    except FileExistsError:
        # Directory already exists, which is fine
        pass
    except Exception as e:
        print(f"Warning: Could not create logs directory: {e}")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log', mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create logger for this application
    logger = logging.getLogger('TicketAnalysisApp')
    logger.setLevel(logging.INFO)
    
    return logger

# Initialize logger
logger = setup_logging()

# Cache configuration
@st.cache_data(ttl=3600)  # Cache for 1 hour
def cache_processed_data(file_content_hash: str, processed_data: pd.DataFrame) -> pd.DataFrame:
    """Cache processed data to avoid reprocessing"""
    return processed_data

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def cache_summary_data(cache_key: str, summary_data: Dict) -> Dict:
    """Cache summary data to avoid regenerating"""
    return summary_data

def generate_cache_key(data: str, model_name: str) -> str:
    """Generate a unique cache key for data and model combination"""
    return hashlib.md5(f"{data}_{model_name}".encode()).hexdigest()

# Page configuration
st.set_page_config(
    page_title="Ticket Data Analysis & Summarization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        font-weight: 700;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid #3498db;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .story-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e3f2fd 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border-left: 5px solid #e74c3c;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .story-section:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        transform: translateY(-1px);
    }
    .log-section {
        background: linear-gradient(135deg, #f4f4f4 0%, #e8e8e8 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border-left: 5px solid #ff6b6b;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        max-height: 300px;
        overflow-y: auto;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .product-group-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-weight: 600;
        font-size: 1.3rem;
    }
    .sidebar-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
    .success-message {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-message {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-message {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stExpander > div > div > div > div {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stExpander > div > div > div > div:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

class TicketDataProcessor:
    """Handles ticket data preprocessing and cleaning"""
    
    def __init__(self):
        self.logger = logging.getLogger('TicketAnalysisApp.TicketDataProcessor')
        self.valid_categories = ['HDW', 'NET', 'KAI', 'KAV', 'GIGA', 'VOD', 'KAD']
        self.category_mapping = {
            'Broadband': ['KAI', 'NET'],
            'Voice': ['KAV'],
            'TV': ['KAD'],
            'GIGA': ['GIGA'],
            'VOD': ['VOD'],
            'Hardware': ['HDW']
        }
        self.logger.info("TicketDataProcessor initialized with valid categories: %s", self.valid_categories)
    
    def _clean_column_name(self, column_name: str) -> str:
        """Clean column name by removing BOM and other encoding artifacts"""
        # Remove BOM character
        cleaned = column_name.replace('\ufeff', '')
        # Remove other common encoding artifacts
        cleaned = cleaned.replace('\u200b', '')  # Zero-width space
        cleaned = cleaned.replace('\u200c', '')  # Zero-width non-joiner
        cleaned = cleaned.replace('\u200d', '')  # Zero-width joiner
        # Strip whitespace
        cleaned = cleaned.strip()
        return cleaned
        
    def parse_ticket_data(self, file_content: str) -> pd.DataFrame:
        """Parse raw ticket data from text file"""
        self.logger.info("Starting to parse ticket data from file content")
        try:
            # Split content into lines and find the header
            lines = file_content.strip().split('\n')
            self.logger.info("File contains %d lines", len(lines))
            
            # Find the header line (first line with commas)
            header_line = None
            data_start = 0
            
            for i, line in enumerate(lines):
                if ',' in line and 'ORDER_NUMBER' in line:
                    header_line = line
                    data_start = i + 1
                    self.logger.info("Found header at line %d: %s", i, header_line[:100] + "..." if len(header_line) > 100 else header_line)
                    break
            
            if header_line is None:
                self.logger.error("Could not find header line in the file")
                st.error("Could not find header line in the file")
                return pd.DataFrame()
            
            # Parse header and clean column names (remove any remaining BOM characters)
            headers = [self._clean_column_name(h) for h in header_line.split(',')]
            self.logger.info("Parsed %d headers: %s", len(headers), headers)
            
            # Parse data rows
            data_rows = []
            for line_num, line in enumerate(lines[data_start:], start=data_start):
                if line.strip() and ',' in line:
                    # Handle quoted fields and split properly
                    row_data = self._parse_csv_line(line)
                    if len(row_data) == len(headers):
                        data_rows.append(row_data)
                    else:
                        self.logger.warning("Line %d has %d fields, expected %d: %s", line_num, len(row_data), len(headers), line[:100])
            
            self.logger.info("Successfully parsed %d data rows", len(data_rows))
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=headers)
            self.logger.info("Created DataFrame with shape: %s", df.shape)
            return df
            
        except Exception as e:
            self.logger.error("Error parsing file: %s", str(e), exc_info=True)
            st.error(f"Error parsing file: {str(e)}")
            return pd.DataFrame()
    
    def _parse_csv_line(self, line: str) -> List[str]:
        """Parse CSV line handling quoted fields"""
        try:
            result = []
            current = ""
            in_quotes = False
            
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    result.append(current.strip())
                    current = ""
                else:
                    current += char
            
            result.append(current.strip())
            return result
        except Exception as e:
            self.logger.error("Error parsing CSV line: %s", str(e), exc_info=True)
            return []
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the ticket data"""
        self.logger.info("Starting data cleaning process for DataFrame with shape: %s", df.shape)
        
        if df.empty:
            self.logger.warning("DataFrame is empty, skipping cleaning")
            return df
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Log initial state
        self.logger.info("Initial data shape: %s", df_clean.shape)
        self.logger.info("Initial columns: %s", list(df_clean.columns))
        self.logger.info("Initial categories: %s", df_clean['SERVICE_CATEGORY'].value_counts().to_dict())
        
        # Filter by valid categories
        before_filter = len(df_clean)
        df_clean = df_clean[df_clean['SERVICE_CATEGORY'].isin(self.valid_categories)]
        after_filter = len(df_clean)
        self.logger.info("Category filtering: %d -> %d rows (removed %d)", before_filter, after_filter, before_filter - after_filter)
        
        # Standardize inconsistent values
        df_clean = self._standardize_values(df_clean)
        
        # Clean datetime columns
        df_clean = self._clean_datetime_columns(df_clean)
        
        # Check if required columns exist before dropna
        required_columns = ['ORDER_NUMBER', 'SERVICE_CATEGORY', 'ACCEPTANCE_TIME']
        missing_columns = [col for col in required_columns if col not in df_clean.columns]
        if missing_columns:
            self.logger.error("Missing required columns: %s", missing_columns)
            self.logger.error("Available columns: %s", list(df_clean.columns))
            raise KeyError(f"Missing required columns: {missing_columns}")
        
        # Remove rows with critical missing data
        before_missing = len(df_clean)
        df_clean = df_clean.dropna(subset=required_columns)
        after_missing = len(df_clean)
        self.logger.info("Missing data removal: %d -> %d rows (removed %d)", before_missing, after_missing, before_missing - after_missing)
        
        # Reset index
        df_clean = df_clean.reset_index(drop=True)
        
        self.logger.info("Final cleaned data shape: %s", df_clean.shape)
        self.logger.info("Final categories: %s", df_clean['SERVICE_CATEGORY'].value_counts().to_dict())
        
        return df_clean
    
    def _standardize_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize inconsistent values in the dataset"""
        self.logger.info("Standardizing values in the dataset")
        
        # Standardize order types
        if 'ORDER_TYPE' in df.columns:
            before_std = df['ORDER_TYPE'].value_counts().to_dict()
            df['ORDER_TYPE'] = df['ORDER_TYPE'].replace({
                'Kurzticket': 'Short Ticket',
                'Aufgabe': 'Task'
            })
            after_std = df['ORDER_TYPE'].value_counts().to_dict()
            self.logger.info("ORDER_TYPE standardization: %s -> %s", before_std, after_std)
        
        # Standardize processing status
        if 'PROCESSING_STATUS' in df.columns:
            before_std = df['PROCESSING_STATUS'].value_counts().to_dict()
            df['PROCESSING_STATUS'] = df['PROCESSING_STATUS'].replace({
                'ab': 'Completed'
            })
            after_std = df['PROCESSING_STATUS'].value_counts().to_dict()
            self.logger.info("PROCESSING_STATUS standardization: %s -> %s", before_std, after_std)
        
        return df
    
    def _clean_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert datetime columns"""
        self.logger.info("Cleaning datetime columns")
        datetime_columns = ['ACCEPTANCE_TIME', 'COMPLETION_TIME', 'CUSTOMER_COMPLETION_TIME']
        
        for col in datetime_columns:
            if col in df.columns:
                self.logger.info("Converting column %s to datetime", col)
                before_convert = df[col].dtype
                df[col] = pd.to_datetime(df[col], errors='coerce', format='%m/%d/%Y %H:%M')
                after_convert = df[col].dtype
                self.logger.info("Column %s conversion: %s -> %s", col, before_convert, after_convert)
                
                # Log conversion success rate
                total_rows = len(df)
                converted_rows = df[col].notna().sum()
                self.logger.info("Column %s conversion success: %d/%d (%.1f%%)", col, converted_rows, total_rows, (converted_rows/total_rows)*100)
        
        return df
    
    def get_product_group(self, service_category: str) -> str:
        """Map service category to product group"""
        product_group = None
        for pg, categories in self.category_mapping.items():
            if service_category in categories:
                product_group = pg
                break
        
        if product_group is None:
            product_group = 'Other'
            self.logger.warning("Service category '%s' mapped to 'Other'", service_category)
        else:
            self.logger.debug("Service category '%s' mapped to '%s'", service_category, product_group)
        
        return product_group

class LLMSummarizer:
    """Handles AI-powered storytelling summaries using Ollama with Llama 3.1 8B"""
    
    def __init__(self, use_ollama: bool = True, model_name: str = "llama3.1:8b"):
        self.logger = logging.getLogger('TicketAnalysisApp.LLMSummarizer')
        self.use_ollama = use_ollama
        self.model_name = model_name
        self.last_raw_response = ""  # Store last raw response for debugging
        
        if self.use_ollama:
            try:
                # Test Ollama connection
                models = ollama.list()
                available_models = [model.model for model in models.models]
                if self.model_name in available_models:
                    self.logger.info("Ollama configured with model: %s", self.model_name)
                else:
                    self.logger.warning("Model %s not found. Available models: %s", self.model_name, available_models)
                    self.logger.warning("Will use fallback summaries")
                    self.use_ollama = False
            except Exception as e:
                self.logger.error("Ollama connection failed: %s", str(e))
                self.logger.warning("Will use fallback summaries")
                self.use_ollama = False
        else:
            self.logger.warning("Ollama disabled, will use fallback summaries")
    
    def get_last_raw_response(self) -> str:
        """Get the last raw LLM response for debugging"""
        return self.last_raw_response
    
    def generate_storytelling_summary(self, product_group: str, tickets: pd.DataFrame) -> Dict:
        """Generate AI-powered storytelling summary for a product group using Ollama"""
        self.logger.info("Generating storytelling summary for product group: %s with %d tickets", product_group, len(tickets))
        
        if not self.use_ollama:
            self.logger.info("Ollama not available, using fallback summary")
            return self._generate_fallback_summary(product_group, tickets)
        
        try:
            # Prepare ticket data for LLM
            self.logger.info("Preparing ticket data for Ollama")
            ticket_summary = self._prepare_ticket_summary(tickets)
            
            # Create prompt for Llama
            self.logger.info("Creating Llama prompt")
            prompt = self._create_summary_prompt(product_group, ticket_summary)
            
            # Call Ollama API
            self.logger.info("Calling Ollama API with model: %s", self.model_name)
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a technical analyst specializing in customer service ticket analysis. Create engaging, structured storytelling summaries based on ticket data."},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 1000
                }
            )
            
            self.logger.info("Ollama API call successful, response received")
            
            # Parse response
            summary_text = response['message']['content']
            self.logger.info("Llama response length: %d characters", len(summary_text))
            
            # Store raw response for debugging
            self.last_raw_response = summary_text
            
            # Structure the response into sections
            parsed_response = self._parse_llm_response(summary_text)
            self.logger.info("Successfully parsed Llama response into %d sections", len(parsed_response))
            
            return parsed_response
            
        except Exception as e:
            self.logger.error("Ollama API call failed: %s", str(e), exc_info=True)
            st.warning(f"Ollama API call failed: {str(e)}. Using fallback summary.")
            return self._generate_fallback_summary(product_group, tickets)
    
    def _prepare_ticket_summary(self, tickets: pd.DataFrame) -> str:
        """Prepare a summary of ticket data for LLM input"""
        self.logger.info("Preparing ticket summary for LLM input")
        
        summary = f"Total tickets: {len(tickets)}\n"
        summary += f"Date range: {tickets['ACCEPTANCE_TIME'].min()} to {tickets['ACCEPTANCE_TIME'].max()}\n"
        summary += f"Categories: {', '.join(tickets['SERVICE_CATEGORY'].unique())}\n"
        summary += f"Status distribution: {tickets['PROCESSING_STATUS'].value_counts().to_dict()}\n"
        
        # Add sample ticket descriptions
        sample_tickets = tickets.head(5)[['ORDER_NUMBER', 'ORDER_DESCRIPTION_1', 'CAUSE', 'COMPLETION_RESULT_KB']].to_string()
        summary += f"\nSample tickets:\n{sample_tickets}"
        
        self.logger.info("Ticket summary prepared, length: %d characters", len(summary))
        return summary
    
    def _create_summary_prompt(self, product_group: str, ticket_summary: str) -> str:
        """Create the prompt for LLM summarization"""
        self.logger.info("Creating summary prompt for product group: %s", product_group)
        
        prompt = f"""
        Analyze the following ticket data for the {product_group} product group and create a structured storytelling summary.
        
        Ticket Data Summary:
        {ticket_summary}
        
        Please create a summary divided into these 5 sections. Use EXACTLY these section titles and format:
        
        Initial Issue:
        [Provide timeframe of first issues, ticket numbers involved, and narrative describing what problems started, how customers reported them, and first resolutions]
        
        Follow-ups:
        [Provide timeframe of follow-up activity, ticket numbers, and narrative about further customer complaints, additional troubleshooting, and technician interventions]
        
        Developments:
        [Provide timeframe of escalations/advancements, ticket numbers, and narrative about new problems, mid-stage fixes, and noticeable improvements]
        
        Later Incidents:
        [Provide timeframe, ticket numbers, and narrative about recurring issues or repeat failures, customer dissatisfaction, and repeated resolutions]
        
        Recent Events:
        [Provide most recent timeframe, ticket numbers, and narrative about latest outcomes, final resolutions, and current customer status]
        
        Make the narrative engaging and business-focused. Include specific ticket numbers and timeframes where possible.
        Ensure each section has substantial content and follows the exact format above.
        """
        
        self.logger.info("Summary prompt created, length: %d characters", len(prompt))
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> Dict:
        """Parse LLM response into structured sections"""
        self.logger.info("Parsing LLM response into structured sections")
        self.logger.info("Raw response preview: %s", response_text[:200] + "..." if len(response_text) > 200 else response_text)
        
        sections = ['Initial Issue', 'Follow-ups', 'Developments', 'Later Incidents', 'Recent Events']
        parsed = {}
        
        for section in sections:
            # Look for section with various possible formats
            section_variants = [
                section,
                f"{section}:",
                f"{section} -",
                f"{section}\n",
                f"{section}\n-",
                f"{section}\n\n"
            ]
            
            found_section = False
            for variant in section_variants:
                if variant in response_text:
                    start_idx = response_text.find(variant)
                    if start_idx != -1:
                        # Find next section or end of text
                        next_section = None
                        next_section_idx = len(response_text)
                        
                        for next_sec in sections:
                            if next_sec != section:
                                # Look for next section with various formats
                                for next_variant in [next_sec, f"{next_sec}:", f"{next_sec} -", f"{next_sec}\n"]:
                                    if next_variant in response_text[start_idx + len(variant):]:
                                        next_idx = response_text.find(next_variant, start_idx + len(variant))
                                        if next_idx != -1 and next_idx < next_section_idx:
                                            next_section_idx = next_idx
                                            break
                        
                        # Extract content between current section and next section
                        content_start = start_idx + len(variant)
                        content = response_text[content_start:next_section_idx].strip()
                        
                        # Clean up the content
                        content = content.strip(':\n- \t')
                        
                        if content:
                            parsed[section] = content
                            self.logger.info("Parsed section '%s': %d characters", section, len(content))
                            found_section = True
                            break
                        
            if not found_section:
                self.logger.warning("Section '%s' not found or empty in response", section)
                parsed[section] = f"Section '{section}' content not available in the LLM response."
        
        self.logger.info("Successfully parsed %d sections from LLM response", len(parsed))
        return parsed
    
    def _generate_fallback_summary(self, product_group: str, tickets: pd.DataFrame) -> Dict:
        """Generate a fallback summary when LLM is not available"""
        self.logger.info("Generating fallback summary for product group: %s", product_group)
        
        if tickets.empty:
            self.logger.warning("No tickets available for fallback summary")
            return {section: "No data available" for section in ['Initial Issue', 'Follow-ups', 'Developments', 'Later Incidents', 'Recent Events']}
        
        # Sort tickets by acceptance time
        sorted_tickets = tickets.sort_values('ACCEPTANCE_TIME')
        
        # Calculate time periods
        total_days = (sorted_tickets['ACCEPTANCE_TIME'].max() - sorted_tickets['ACCEPTANCE_TIME'].min()).days
        period_size = max(1, total_days // 5)
        
        self.logger.info("Fallback summary: total_days=%d, period_size=%d", total_days, period_size)
        
        sections = {}
        for i, section in enumerate(['Initial Issue', 'Follow-ups', 'Developments', 'Later Incidents', 'Recent Events']):
            start_date = sorted_tickets['ACCEPTANCE_TIME'].min() + timedelta(days=i * period_size)
            end_date = start_date + timedelta(days=period_size)
            
            period_tickets = sorted_tickets[
                (sorted_tickets['ACCEPTANCE_TIME'] >= start_date) & 
                (sorted_tickets['ACCEPTANCE_TIME'] < end_date)
            ]
            
            if not period_tickets.empty:
                ticket_numbers = ', '.join(period_tickets['ORDER_NUMBER'].head(3).tolist())
                main_issues = period_tickets['ORDER_DESCRIPTION_1'].value_counts().head(2).index.tolist()
                
                sections[section] = f"Timeframe: {start_date.strftime('%m/%d/%Y')} - {end_date.strftime('%m/%d/%Y')}\n"
                sections[section] += f"Ticket numbers: {ticket_numbers}\n"
                sections[section] += f"Main issues: {', '.join(main_issues)}"
                
                self.logger.info("Fallback section '%s': %d tickets, timeframe %s - %s", 
                               section, len(period_tickets), start_date.strftime('%m/%d/%Y'), end_date.strftime('%m/%d/%Y'))
            else:
                sections[section] = f"No tickets in this period"
                self.logger.info("Fallback section '%s': no tickets in period", section)
        
        self.logger.info("Fallback summary generated with %d sections", len(sections))
        return sections

class DataVisualizer:
    """Handles all data visualizations and charts"""
    
    def __init__(self):
        self.logger = logging.getLogger('TicketAnalysisApp.DataVisualizer')
        self.logger.info("DataVisualizer initialized")
    
    def plot_trend_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Create trend analysis chart showing tickets over time"""
        self.logger.info("Creating trend analysis chart")
        
        if df.empty:
            self.logger.warning("DataFrame is empty, returning empty figure")
            return go.Figure()
        
        # Group by date and count tickets
        daily_counts = df.groupby(df['ACCEPTANCE_TIME'].dt.date).size().reset_index()
        daily_counts.columns = ['Date', 'Ticket Count']
        
        self.logger.info("Trend analysis: %d data points", len(daily_counts))
        
        fig = px.line(
            daily_counts, 
            x='Date', 
            y='Ticket Count',
            title='Daily Ticket Volume Over Time',
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Tickets",
            hovermode='x unified'
        )
        
        self.logger.info("Trend analysis chart created successfully")
        return fig
    
    def plot_category_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create category distribution chart"""
        self.logger.info("Creating category distribution chart")
        
        if df.empty:
            self.logger.warning("DataFrame is empty, returning empty figure")
            return go.Figure()
        
        category_counts = df['SERVICE_CATEGORY'].value_counts()
        self.logger.info("Category distribution: %s", category_counts.to_dict())
        
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title='Ticket Distribution by Service Category'
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        self.logger.info("Category distribution chart created successfully")
        return fig
    
    def plot_resolution_times(self, df: pd.DataFrame) -> go.Figure:
        """Create resolution time analysis chart"""
        self.logger.info("Creating resolution time analysis chart")
        
        if df.empty:
            self.logger.warning("DataFrame is empty, returning empty figure")
            return go.Figure()
        
        # Calculate resolution time
        df_with_time = df.copy()
        df_with_time['Resolution_Time_Minutes'] = (
            df_with_time['COMPLETION_TIME'] - df_with_time['ACCEPTANCE_TIME']
        ).dt.total_seconds() / 60
        
        # Group by category and calculate average
        avg_resolution = df_with_time.groupby('SERVICE_CATEGORY')['Resolution_Time_Minutes'].mean().reset_index()
        self.logger.info("Average resolution times: %s", avg_resolution.to_dict('records'))
        
        fig = px.bar(
            avg_resolution,
            x='SERVICE_CATEGORY',
            y='Resolution_Time_Minutes',
            title='Average Resolution Time by Service Category (Minutes)'
        )
        
        fig.update_layout(
            xaxis_title="Service Category",
            yaxis_title="Average Resolution Time (Minutes)"
        )
        
        self.logger.info("Resolution time chart created successfully")
        return fig
    
    def plot_customer_insights(self, df: pd.DataFrame) -> go.Figure:
        """Create customer insights chart"""
        self.logger.info("Creating customer insights chart")
        
        if df.empty:
            self.logger.warning("DataFrame is empty, returning empty figure")
            return go.Figure()
        
        customer_ticket_counts = df['CUSTOMER_NUMBER'].value_counts().head(10)
        self.logger.info("Top 10 customers by ticket count: %s", customer_ticket_counts.to_dict())
        
        fig = px.bar(
            x=customer_ticket_counts.index.astype(str),
            y=customer_ticket_counts.values,
            title='Top 10 Customers by Ticket Count',
            labels={'x': 'Customer Number', 'y': 'Number of Tickets'}
        )
        
        fig.update_layout(
            xaxis_title="Customer Number",
            yaxis_title="Number of Tickets"
        )
        
        self.logger.info("Customer insights chart created successfully")
        return fig
    
    def plot_cause_resolution_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Create cause and resolution analysis chart"""
        self.logger.info("Creating cause and resolution analysis chart")
        
        if df.empty:
            self.logger.warning("DataFrame is empty, returning empty figure")
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Top Failure Causes', 'Top Resolution Methods'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Top causes
        top_causes = df['CAUSE'].value_counts().head(8)
        self.logger.info("Top failure causes: %s", top_causes.to_dict())
        
        fig.add_trace(
            go.Bar(x=top_causes.values, y=top_causes.index, orientation='h', name='Causes'),
            row=1, col=1
        )
        
        # Top resolutions
        top_resolutions = df['COMPLETION_RESULT_KB'].value_counts().head(8)
        self.logger.info("Top resolution methods: %s", top_resolutions.to_dict())
        
        fig.add_trace(
            go.Bar(x=top_resolutions.values, y=top_resolutions.index, orientation='h', name='Resolutions'),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Failure Causes vs Resolution Methods Analysis",
            height=500
        )
        
        self.logger.info("Cause and resolution analysis chart created successfully")
        return fig

def main():
    """Main Streamlit application"""
    
    logger.info("Starting Ticket Data Analysis & Summarization application")
    
    # Header
    st.markdown('<h1 class="main-header">🎫 Ticket Data Analysis & Summarization</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Ticket Data File",
        type=['txt', 'csv'],
        help="Upload your ticket data file (TXT or CSV format)"
    )
    
    # Ollama configuration
    use_ollama = st.sidebar.checkbox(
        "Use Ollama (Llama 3.1 8B)",
        value=True,
        help="Enable local LLM processing using Ollama with Llama 3.1 8B"
    )
    
    # Get available models from Ollama
    available_models = ["llama3.1:8b"]  # Default fallback
    if use_ollama:
        try:
            models = ollama.list()
            available_models = [model.model for model in models.models]
        except Exception as e:
            st.sidebar.warning(f"Could not fetch Ollama models: {str(e)}")
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Ollama Model",
        options=available_models,
        index=0,
        help="Choose which Ollama model to use for AI summaries"
    )
    
    # Debug options
    show_logs = st.sidebar.checkbox("Show Debug Logs", value=False, help="Display application logs in the main interface")
    
    # Cache management
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗄️ Cache Management")
    
    # Show cache status
    cache_info = st.sidebar.container()
    with cache_info:
        st.write("**Cache Status:**")
        if 'processed_data' in st.session_state:
            st.success("✅ Data cached")
        else:
            st.info("📋 No data cached")
        
        summary_cache_count = sum(1 for key in st.session_state.keys() if key.startswith('summary_'))
        if summary_cache_count > 0:
            st.success(f"✅ {summary_cache_count} summaries cached")
        else:
            st.info("📋 No summaries cached")
    
    # Clear cache button
    if st.sidebar.button("🗑️ Clear All Cache", help="Remove all cached data and summaries"):
        # Clear all cache keys
        keys_to_remove = [key for key in st.session_state.keys() if key.startswith(('processed_', 'summary_'))]
        for key in keys_to_remove:
            del st.session_state[key]
        st.sidebar.success("Cache cleared successfully!")
        st.rerun()
    
    # Initialize classes
    processor = TicketDataProcessor()
    summarizer = LLMSummarizer(use_ollama=use_ollama, model_name=model_name)
    visualizer = DataVisualizer()
    
    # Main content
    if uploaded_file is not None:
        logger.info("File uploaded: %s (size: %d bytes)", uploaded_file.name, uploaded_file.size)
        
        try:
            # Read file content
            if uploaded_file.name.endswith('.txt'):
                content = uploaded_file.read().decode('utf-8')
                logger.info("Reading TXT file content, length: %d characters", len(content))
                
                # Generate cache key for this file
                file_hash = hashlib.md5(content.encode()).hexdigest()
                cache_key = f"processed_{file_hash}"
                
                # Check if we have cached processed data
                if cache_key in st.session_state:
                    logger.info("Using cached processed data")
                    df_clean = st.session_state[cache_key]
                    st.success(f"✅ Loaded {len(df_clean)} tickets from cache")
                else:
                    logger.info("Processing file for the first time")
                    df = processor.parse_ticket_data(content)
                    
                    if not df.empty:
                        logger.info("File parsed successfully, DataFrame shape: %s", df.shape)
                        
                        # Clean data
                        df_clean = processor.clean_data(df)
                        
                        if not df_clean.empty:
                            # Cache the processed data
                            st.session_state[cache_key] = df_clean
                            logger.info("Processed data cached for future use")
                            st.success(f"✅ Successfully loaded and cleaned {len(df_clean)} tickets")
                        else:
                            logger.error("No valid tickets found after cleaning")
                            st.error("❌ No valid tickets found after cleaning. Please check your data format.")
                            return
                    else:
                        logger.error("Failed to parse the uploaded file")
                        st.error("❌ Failed to parse the uploaded file. Please check the file format.")
                        return
            else:
                logger.info("Reading CSV file")
                df = pd.read_csv(uploaded_file)
                
                if not df.empty:
                    logger.info("File parsed successfully, DataFrame shape: %s", df.shape)
                    
                    # Clean data
                    df_clean = processor.clean_data(df)
                    
                    if not df_clean.empty:
                        st.success(f"✅ Successfully loaded and cleaned {len(df_clean)} tickets")
                    else:
                        logger.error("No valid tickets found after cleaning")
                        st.error("❌ No valid tickets found after cleaning. Please check your data format.")
                        return
                else:
                    logger.error("Failed to parse the uploaded file")
                    st.error("❌ Failed to parse the uploaded file. Please check the file format.")
                    return
                
            # Display data overview
            st.markdown('<h2 class="section-header">📊 Data Overview</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Tickets", len(df_clean))
            
            with col2:
                st.metric("Service Categories", len(df_clean['SERVICE_CATEGORY'].unique()))
            
            with col3:
                st.metric("Customers", len(df_clean['CUSTOMER_NUMBER'].unique()))
            
            with col4:
                avg_time = (df_clean['COMPLETION_TIME'] - df_clean['ACCEPTANCE_TIME']).dt.total_seconds().mean() / 60
                st.metric("Avg Resolution Time", f"{avg_time:.1f} min")
            
            # Add product group column
            df_clean['PRODUCT_GROUP'] = df_clean['SERVICE_CATEGORY'].apply(processor.get_product_group)
            
            # Display cleaned data
            st.markdown('<h3>Cleaned Dataset Preview</h3>', unsafe_allow_html=True)
            st.dataframe(df_clean.head(10))
            
            # Storytelling Summaries
            st.markdown('<h2 class="section-header">🤖 AI-Powered Storytelling Summaries</h2>', unsafe_allow_html=True)
            
            # Generate summaries for each product group
            product_groups = df_clean['PRODUCT_GROUP'].unique()
            logger.info("Generating summaries for product groups: %s", list(product_groups))
            
            for product_group in product_groups:
                if product_group != 'Other':
                    st.markdown(f'<div class="product-group-header">🎯 {product_group} Product Group</div>', unsafe_allow_html=True)
                    
                    # Filter tickets for this product group
                    group_tickets = df_clean[df_clean['PRODUCT_GROUP'] == product_group]
                    logger.info("Processing %s group with %d tickets", product_group, len(group_tickets))
                    
                    # Generate cache key for summary
                    summary_cache_key = f"summary_{file_hash}_{product_group}_{model_name}"
                    
                    # Check if we have cached summary
                    if summary_cache_key in st.session_state:
                        logger.info("Using cached summary for %s", product_group)
                        summary = st.session_state[summary_cache_key]
                        st.info(f"📋 Using cached summary for {product_group}")
                    else:
                        logger.info("Generating new summary for %s", product_group)
                        # Generate summary
                        summary = summarizer.generate_storytelling_summary(product_group, group_tickets)
                        
                        # Cache the summary
                        st.session_state[summary_cache_key] = summary
                        logger.info("Summary cached for %s", product_group)
                    
                    # Debug logging (kept for backend debugging, not shown in UI)
                    logger.info("Summary generated for %s: %s", product_group, list(summary.keys()))
                    for section_name, section_content in summary.items():
                        logger.info("Section '%s' content length: %d", section_name, len(str(section_content)))
                        logger.info("Section '%s' content preview: %s", section_name, str(section_content)[:100] + "..." if len(str(section_content)) > 100 else str(section_content))
                    
                    # Display summary sections
                    for section_name, section_content in summary.items():
                        with st.expander(f"📖 {section_name}", expanded=True):
                            if section_content and str(section_content).strip():
                                # Use simple text display instead of HTML markdown
                                st.write(section_content)
                            else:
                                st.warning(f"No content available for {section_name}")
                                st.info("This may indicate an issue with the LLM response parsing.")
            
            # Visualizations
            st.markdown('<h2 class="section-header">📈 Data Visualizations</h2>', unsafe_allow_html=True)
            
            # Trend Analysis
            st.subheader("Trend Analysis")
            trend_fig = visualizer.plot_trend_analysis(df_clean)
            st.plotly_chart(trend_fig, use_container_width=True)
            
            # Category Distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Category Distribution")
                category_fig = visualizer.plot_category_distribution(df_clean)
                st.plotly_chart(category_fig, use_container_width=True)
            
            with col2:
                st.subheader("Resolution Times")
                resolution_fig = visualizer.plot_resolution_times(df_clean)
                st.plotly_chart(resolution_fig, use_container_width=True)
            
            # Customer Insights
            st.subheader("Customer Insights")
            customer_fig = visualizer.plot_customer_insights(df_clean)
            st.plotly_chart(customer_fig, use_container_width=True)
            
            # Cause and Resolution Analysis
            st.subheader("Failure Causes & Resolution Methods")
            cause_resolution_fig = visualizer.plot_cause_resolution_analysis(df_clean)
            st.plotly_chart(cause_resolution_fig, use_container_width=True)
            
            # Bonus Insights
            st.markdown('<h2 class="section-header">🔍 Bonus Insights & Business Intelligence</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Trend Analysis")
                
                # Most recurring problems
                recurring_issues = df_clean['ORDER_DESCRIPTION_1'].value_counts().head(5)
                st.write("**Most Common Issues:**")
                for issue, count in recurring_issues.items():
                    st.write(f"• {issue}: {count} tickets")
                
                # Product category with most problems
                category_problems = df_clean['SERVICE_CATEGORY'].value_counts()
                st.write(f"\n**Category with Most Issues:** {category_problems.index[0]} ({category_problems.iloc[0]} tickets)")
            
            with col2:
                st.subheader("Customer Risk Analysis")
                
                # Customers with repeated complaints
                customer_ticket_counts = df_clean['CUSTOMER_NUMBER'].value_counts()
                heavy_users = customer_ticket_counts[customer_ticket_counts > 2]
                
                if not heavy_users.empty:
                    st.write("**Customers with 3+ Tickets (Churn Risk):**")
                    for customer, count in heavy_users.head(5).items():
                        st.write(f"• Customer {customer}: {count} tickets")
                else:
                    st.write("No customers with 3+ tickets identified")
                
                # Business Improvement Suggestions
                st.subheader("Business Improvement Suggestions")
                
                suggestions = []
                
                # HDW suggestions
                hdw_tickets = df_clean[df_clean['SERVICE_CATEGORY'] == 'HDW']
                if len(hdw_tickets) > 5:
                    suggestions.append("**HDW (Hardware):** Consider proactive router replacement program for customers with multiple hardware issues")
                
                # NET suggestions
                net_tickets = df_clean[df_clean['SERVICE_CATEGORY'] == 'NET']
                if len(net_tickets) > 3:
                    suggestions.append("**NET (Internet):** Implement bandwidth monitoring to detect issues before customer complaints")
                
                # KAI suggestions
                kai_tickets = df_clean[df_clean['SERVICE_CATEGORY'] == 'KAI']
                if len(kai_tickets) > 2:
                    suggestions.append("**KAI (Smartcard):** Provide self-service PIN reset options to reduce ticket volume")
                
                if suggestions:
                    for suggestion in suggestions:
                        st.write(f"💡 {suggestion}")
                else:
                    st.write("No specific improvement suggestions based on current data volume")
                
                # Debug Logs Section
                if show_logs:
                    st.markdown('<h2 class="section-header">🔍 Debug Logs</h2>', unsafe_allow_html=True)
                    
                    # Read and display recent logs
                    try:
                        with open('logs/app.log', 'r') as log_file:
                            log_lines = log_file.readlines()
                            # Show last 50 lines
                            recent_logs = log_lines[-50:] if len(log_lines) > 50 else log_lines
                            log_content = ''.join(recent_logs)
                            
                            st.markdown('<div class="log-section">', unsafe_allow_html=True)
                            st.text(log_content)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Download logs option
                            st.download_button(
                                label="Download Full Logs",
                                data=''.join(log_lines),
                                file_name=f"ticket_analysis_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                                mime="text/plain"
                            )
                    except FileNotFoundError:
                        st.warning("No log file found. Logs will appear here after processing data.")
                    except Exception as e:
                        st.error(f"Error reading logs: {str(e)}")
                 
        except Exception as e:
            logger.error("Error processing file: %s", str(e), exc_info=True)
            st.error(f"❌ Error processing file: {str(e)}")
            
            # Show detailed error in debug mode
            if show_logs:
                st.markdown('<h3>Detailed Error Information</h3>', unsafe_allow_html=True)
                st.markdown('<div class="log-section">', unsafe_allow_html=True)
                st.text(traceback.format_exc())
                st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Welcome message and instructions
        st.markdown("""
        ## Welcome to the Ticket Data Analysis & Summarization Tool! 🎯
        
        This application helps you analyze customer service ticket data and generate AI-powered storytelling summaries using **Ollama with Llama 3.1 8B**.
        
        ### How to get started:
        1. **Upload your ticket data file** using the sidebar (TXT or CSV format)
        2. **Enable Ollama** for AI-powered summaries using local LLM
        3. **Select your preferred model** (Llama 3.1 8B recommended)
        4. **Enable debug logs** to see detailed processing information
        5. **View the analysis results** including:
           - Data overview and statistics
           - AI-generated storytelling summaries by product group
           - Interactive visualizations and charts
           - Business intelligence insights
           - Improvement suggestions
        
        ### Supported file formats:
        - Text files (.txt) with comma-separated values
        - CSV files (.csv)
        
        ### Features:
        - ✅ Data preprocessing and cleaning
        - ✅ Category mapping to business product groups
        - ✅ **Local AI-powered storytelling summaries** (Ollama + Llama 3.1 8B)
        - ✅ Interactive visualizations
        - ✅ Trend analysis and insights
        - ✅ Customer risk assessment
        - ✅ Business improvement suggestions
        - ✅ Comprehensive logging and debugging
        - ✅ **Offline LLM processing** (no API keys needed!)
        
        Upload a file to begin your analysis!
        """)
        
        # Sample data preview
        st.markdown("### Sample Data Format")
        st.info("""
        Your data should include columns like:
        - ORDER_NUMBER
        - SERVICE_CATEGORY (HDW, NET, KAI, KAV, GIGA, VOD, KAD)
        - ACCEPTANCE_TIME
        - COMPLETION_TIME
        - CUSTOMER_NUMBER
        - ORDER_DESCRIPTION_1
        - CAUSE
        - COMPLETION_RESULT_KB
        """)
        
        # Log file location info
        if show_logs:
            st.markdown("### Log Files")
            st.info("""
            Application logs are stored in the `logs/` directory:
            - `logs/app.log` - Main application log file
            - Logs include detailed information about data processing, API calls, and errors
            - Enable "Show Debug Logs" to view logs in the application
            """)
            try:
                with open('logs/app.log', 'r') as f:
                    log_content = f.read()
                    st.markdown('<h3>Application Logs</h3>', unsafe_allow_html=True)
                    st.text(log_content)
            except FileNotFoundError:
                st.warning("No log file found. Logs will appear here after processing data.")
            except Exception as e:
                st.error(f"Error reading logs: {str(e)}")

if __name__ == "__main__":
    main()
