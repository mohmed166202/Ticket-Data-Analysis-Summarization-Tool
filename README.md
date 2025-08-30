# 🎫 Ticket Data Analysis & Summarization Tool

A comprehensive Streamlit web application for analyzing customer service ticket data and generating AI-powered storytelling summaries. This tool provides data preprocessing, category mapping, interactive visualizations, and business intelligence insights.

## 🚀 Features

- **Data Preprocessing**: Upload and clean raw ticket data from TXT/CSV files
- **Category Mapping**: Automatically map service categories to business product groups
- **AI-Powered Summaries**: Generate engaging storytelling summaries using OpenAI GPT API
- **Interactive Visualizations**: Trend analysis, category distribution, resolution times, and more
- **Business Intelligence**: Identify patterns, customer risks, and improvement opportunities
- **Modular Architecture**: Clean, maintainable code structure with comprehensive documentation

## 📋 Requirements

- Python 3.8+
- Ollama installed locally with Llama 3.1 8B model
- Internet connection for package installation

## 🛠️ Installation

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd ticket-data-analysis
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Ollama with Llama 3.1 8B**
   ```bash
   # Install Ollama (if not already installed)
   # Visit: https://ollama.ai/download
   
   # Pull the Llama 3.1 8B model
   ollama pull llama3.1:8b
   
   # Verify installation
   ollama list
   ```

## 🚀 Usage

### Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the displayed URL (usually `http://localhost:8501`)

3. **Upload your ticket data file** using the sidebar
   - Supported formats: `.txt` (comma-separated) or `.csv`
   - File should contain columns like: `ORDER_NUMBER`, `SERVICE_CATEGORY`, `ACCEPTANCE_TIME`, etc.

4. **Ensure Ollama is running** with Llama 3.1 8B model for AI-powered summaries

5. **Explore the analysis results**:
   - Data overview and statistics
   - AI-generated storytelling summaries by product group
   - Interactive visualizations and charts
   - Business intelligence insights
   - Improvement suggestions
   - **NEW**: Enhanced UI with modern styling and hover effects
   - **NEW**: Smart caching for faster repeated analyses
   - **NEW**: Hardware (HDW) category properly mapped and analyzed

## 📊 Data Format Requirements

Your ticket data should include these essential columns:

| Column | Description | Example |
|--------|-------------|---------|
| `ORDER_NUMBER` | Unique ticket identifier | `001-0671177/24` |
| `SERVICE_CATEGORY` | Service category code | `HDW`, `NET`, `KAI`, `KAV`, `GIGA`, `VOD`, `KAD` |
| `ACCEPTANCE_TIME` | Ticket creation time | `11/05/2024 11:00` |
| `COMPLETION_TIME` | Ticket resolution time | `11/05/2024 11:30` |
| `CUSTOMER_NUMBER` | Customer identifier | `123` |
| `ORDER_DESCRIPTION_1` | Issue description | `Cable Router`, `WLAN`, `Slow` |
| `CAUSE` | Root cause of issue | `URS_KIP_Reset_WLAN_Settings` |
| `COMPLETION_RESULT_KB` | Resolution method | `WLAN settings optimized` |

## 🏗️ Implementation Architecture

### 1. Data Preprocessing (`TicketDataProcessor`)

The application processes raw ticket data through several stages:

- **File Parsing**: Handles both TXT and CSV formats with proper CSV line parsing
- **Data Cleaning**: Filters valid categories, standardizes values, and handles missing data
- **Category Mapping**: Maps service categories to business product groups:
  - **Broadband**: KAI, NET
  - **Voice**: KAV
  - **TV**: KAD
  - **GIGA**: GIGA
  - **VOD**: VOD

### 2. AI-Powered Summarization (`LLMSummarizer`)

Generates structured storytelling summaries for each product group:

- **Structured Output**: Divides summaries into 5 sections:
  1. Initial Issue
  2. Follow-ups
  3. Developments
  4. Later Incidents
  5. Recent Events
- **Fallback Mode**: Provides automated summaries when LLM is unavailable

### 3. Data Visualization (`DataVisualizer`)

Creates interactive charts and dashboards for business intelligence:

- **Trend Analysis**: Daily ticket volume over time with interactive Plotly charts
- **Category Distribution**: Pie charts showing ticket distribution across service categories
- **Resolution Times**: Performance benchmarking by category
- **Customer Insights**: Identification of high-risk customers and churn patterns
- **Failure Analysis**: Root cause analysis and resolution method distribution

### 4. Performance & UI Enhancements

**Smart Caching System:**
- **Data Caching**: Stores processed data to avoid reprocessing the same files
- **Summary Caching**: Caches LLM-generated summaries for faster repeated access
- **Cache Management**: User-friendly cache status display and manual cache clearing
- **Performance Boost**: Significantly faster subsequent analyses of the same data

**Enhanced User Interface:**
- **Modern Design**: Gradient backgrounds, hover effects, and smooth transitions
- **Responsive Layout**: Better visual hierarchy and improved readability
- **Professional Styling**: Enhanced cards, sections, and interactive elements
- **Visual Feedback**: Hover animations and improved color schemes

**Improved Category Mapping:**
- **HDW Classification**: Now properly mapped to 'Hardware' product group instead of 'Other'
- **Better Organization**: More logical grouping for business analysis
- **Enhanced Insights**: Hardware-specific business improvement suggestions

### 5. Business Intelligence

The application provides actionable business insights:

- **Trend Identification**: Most recurring problems and problematic product categories
- **Customer Risk Analysis**: Identification of customers with multiple tickets (churn risk)
- **Business Improvements**: Specific suggestions for each service category:
  - **HDW (Hardware)**: Proactive router replacement programs
  - **NET (Internet)**: Bandwidth monitoring implementation
  - **KAI (Smartcard)**: Self-service PIN reset options
- **Operational Efficiency**: Data-driven recommendations for service improvement

## 🔧 Code Documentation

### Class: `TicketDataProcessor`

Handles all data preprocessing and cleaning operations.

**Methods:**
- `parse_ticket_data(file_content)`: Parses raw text file content into DataFrame
- `clean_data(df)`: Cleans and preprocesses ticket data
- `get_product_group(service_category)`: Maps service category to business product group

### Class: `LLMSummarizer`

Manages AI-powered storytelling summary generation.

**Methods:**
- `generate_storytelling_summary(product_group, tickets)`: Main method for generating summaries
- `_prepare_ticket_summary(tickets)`: Prepares ticket data for LLM input
- `_create_summary_prompt(product_group, ticket_summary)`: Creates LLM prompts
- `_generate_fallback_summary(product_group, tickets)`: Fallback when LLM unavailable

### Class: `DataVisualizer`

Creates all charts and visualizations.

**Methods:**
- `plot_trend_analysis(df)`: Daily ticket volume trends
- `plot_category_distribution(df)`: Service category distribution
- `plot_resolution_times(df)`: Average resolution times by category
- `plot_customer_insights(df)`: Customer ticket count analysis
- `plot_cause_resolution_analysis(df)`: Failure causes vs resolution methods

## 📈 Business Intelligence Features

### Trend Analysis
- Identifies peak ticket periods
- Tracks issue escalation patterns
- Monitors resolution time trends

### Customer Risk Assessment
- Flags customers with 3+ tickets (potential churn risk)
- Identifies heavy users requiring special attention
- Tracks customer satisfaction through resolution success rates

### Product Performance
- Compares resolution times across service categories
- Identifies most problematic product areas
- Suggests proactive maintenance strategies

### Improvement Opportunities
- **HDW (Hardware)**: Proactive router replacement programs
- **NET (Internet)**: Bandwidth monitoring and early detection
- **KAI (Smartcard)**: Self-service PIN reset options
- **KAV (Voice)**: Signal quality monitoring
- **VOD (Video)**: Content delivery optimization



## 🚀 Future Enhancements

- **Real-time Data Integration**: Connect to live ticket systems
- **Advanced Analytics**: Machine learning for predictive maintenance
- **Custom Dashboards**: User-configurable visualization layouts
- **Export Functionality**: PDF reports and data exports
- **Multi-language Support**: Internationalization for global teams

## 📞 Support

For technical support or feature requests:

1. Check the troubleshooting section above
2. Review the code documentation
3. Open an issue in the repository
4. Contact the development team

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ using Streamlit, Pandas, Plotly, Ollama**
