# 🚀 Quick Start Guide

Get up and running with the Ticket Data Analysis & Summarization Tool in under 5 minutes!

## ⚡ Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Ollama with Llama 3.1 8B
```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai/download

# Pull the Llama 3.1 8B model
ollama pull llama3.1:8b

# Verify installation
ollama list
```

### 3. Run the Application
```bash
streamlit run app.py
```

### 4. Open Your Browser
Navigate to `http://localhost:8501`

### 5. Upload Your Data
- Use the sidebar to upload your ticket data file
- Supported formats: `.txt` or `.csv`
- Ensure Ollama is running for AI-powered summaries

## 📁 Sample Data Format

Your data should look like this:

```csv
ORDER_NUMBER,SERVICE_CATEGORY,ACCEPTANCE_TIME,COMPLETION_TIME,CUSTOMER_NUMBER,ORDER_DESCRIPTION_1,CAUSE,COMPLETION_RESULT_KB
001-0671177/24,HDW,11/05/2024 11:00,11/05/2024 11:30,123,Cable Router,URS_KIP_Reset_WLAN_Settings,WLAN settings optimized
001-0682295/24,NET,11/05/2024 13:00,11/05/2024 13:45,123,Request - KIP,CM_BERA_KD_Complaint_about_BI_Installation,Technician appointment scheduled
```

## 🎯 What You'll Get

### ✅ Data Overview
- Total tickets, categories, customers, and resolution times
- Cleaned dataset preview with enhanced UI

### ✅ AI-Powered Summaries
- Storytelling summaries for each product group using local Llama 3.1 8B
- 5 structured sections: Initial Issue, Follow-ups, Developments, Later Incidents, Recent Events
- **NEW**: Hardware (HDW) category properly mapped and analyzed

### ✅ Interactive Visualizations
- Trend analysis charts with enhanced styling
- Category distribution pie charts
- Resolution time comparisons
- Customer insights with modern design

### ✅ Business Intelligence
- Pattern recognition and trend analysis
- Customer risk assessment (churn detection)
- **NEW**: Hardware-specific improvement suggestions
- **NEW**: Enhanced business recommendations

### ✅ **NEW: Enhanced Features**
- **Smart Caching**: Faster repeated analyses with cache management
- **Modern UI**: Gradient backgrounds, hover effects, and smooth animations
- **Performance Monitoring**: Real-time cache status and performance metrics
- **Professional Design**: Business-ready interface with enhanced visual hierarchy

## 🔧 Configuration

### Ollama Setup
- **Local Processing**: No API costs or internet dependency
- **Model Selection**: Llama 3.1 8B for optimal performance
- **Automatic Fallback**: Graceful degradation when Ollama unavailable

### File Upload
- **Maximum file size**: 200MB (increased from 50MB)
- **Supported formats**: TXT, CSV with automatic format detection
- **Automatic data cleaning**: BOM removal, encoding handling, validation

### **NEW: Cache Management**
- **Data Caching**: Stores processed data for 1 hour
- **Summary Caching**: Stores LLM summaries for 30 minutes
- **Cache Control**: Manual cache clearing and status monitoring
- **Performance Boost**: 80-90% faster on repeated analyses

## 🚨 Troubleshooting

### Common Issues

**File won't upload?**
- Check file format (TXT or CSV)
- Ensure file size < 200MB
- Verify file encoding (UTF-8 recommended)

**Charts not loading?**
- Refresh the page
- Check browser console for errors
- Ensure data contains valid datetime values

**LLM summaries not working?**
- Verify Ollama is running: `ollama list`
- Check if Llama 3.1 8B model is available
- Ensure Ollama service is accessible at localhost:11434

**Performance issues?**
- Check cache status in sidebar
- Clear cache if needed
- Restart Ollama service if summaries are slow

### Performance Tips
- **Use caching**: Subsequent analyses are much faster
- **Close other tabs**: Free up browser memory
- **Monitor cache**: Check cache status for optimal performance
- **Restart if needed**: Restart app if it becomes slow

## 📚 Next Steps

1. **Explore the Data**: Upload your ticket data and explore the enhanced visualizations
2. **Generate Summaries**: Use local LLM-powered storytelling for business insights
3. **Analyze Patterns**: Identify trends and customer risks with improved UI
4. **Get Recommendations**: Review enhanced business improvement suggestions
5. **Monitor Performance**: Use cache management for optimal experience

## 🆘 Need Help?

- Check the [README.md](README.md) for detailed documentation
- Review the [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for technical details
- Open an issue in the repository for support

## 🆕 **What's New in This Version?**

- **Ollama Integration**: Local LLM processing instead of OpenAI API
- **HDW Hardware Mapping**: Proper classification of hardware tickets
- **Smart Caching System**: 80-90% performance improvement
- **Enhanced UI**: Modern design with gradients, hover effects, and animations
- **Cache Management**: User-friendly cache control and monitoring
- **Increased File Size**: Support for files up to 200MB

---

**Ready to analyze your ticket data with enhanced performance and beautiful UI? Start the app and upload your file! 🎉**
