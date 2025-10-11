#!/usr/bin/env python3
"""
ML Specialization Assignment Generator
Creates all 25 assignments for the ML specialization track
"""

import os
import json

# Assignment configuration data
ASSIGNMENTS_CONFIG = [
    {
        "id": 6,
        "title": "Model Evaluation & Regularization",
        "subtitle": "Master overfitting/underfitting concepts and comprehensive evaluation",
        "difficulty": "Intermediate",
        "duration": "5-7 hours",
        "technologies": ["Cross-validation", "Regularization", "Metrics", "Validation"],
        "business_scenario": {
            "company": "MedTech Analytics",
            "challenge": "Build robust medical diagnostic models that generalize well to new patients and different hospitals",
            "goals": ["Prevent overfitting to training data", "Ensure model reliability", "Implement proper validation"]
        },
        "phases": [
            "Cross-validation strategies",
            "Regularization techniques", 
            "Performance metrics",
            "Model selection"
        ]
    },
    {
        "id": 7,
        "title": "ML vs DL Comparison Project",
        "subtitle": "Mini-project comparing traditional ML and deep learning approaches",
        "difficulty": "Intermediate", 
        "duration": "8-10 hours",
        "technologies": ["Comparison", "Analysis", "Benchmarking", "Reporting"],
        "business_scenario": {
            "company": "DataFlow Consulting",
            "challenge": "Provide clients with data-driven recommendations on when to use traditional ML vs deep learning",
            "goals": ["Comprehensive comparison framework", "Cost-benefit analysis", "Implementation guidelines"]
        },
        "phases": [
            "Dataset preparation",
            "ML model implementation",
            "DL model implementation", 
            "Comparative analysis"
        ]
    },
    {
        "id": 8,
        "title": "CNN, RNN & LSTM Deep Dive",
        "subtitle": "Implement Convolutional and Recurrent Neural Networks",
        "difficulty": "Advanced",
        "duration": "7-9 hours", 
        "technologies": ["CNN", "RNN", "LSTM", "Computer Vision"],
        "business_scenario": {
            "company": "VisionTech AI",
            "challenge": "Build specialized neural networks for image recognition and time series prediction",
            "goals": ["Master CNN architectures", "Understand sequential data processing", "Compare network types"]
        },
        "phases": [
            "CNN implementation",
            "RNN fundamentals",
            "LSTM architecture",
            "Performance comparison"
        ]
    },
    {
        "id": 9,
        "title": "Advanced DL Optimization",
        "subtitle": "Explore advanced optimizers and hyperparameter tuning",
        "difficulty": "Advanced",
        "duration": "6-8 hours",
        "technologies": ["Adam", "Hyperparameter Tuning", "Learning Rate", "Optimization"],
        "business_scenario": {
            "company": "OptimalAI Research",
            "challenge": "Develop next-generation optimization strategies for complex deep learning models",
            "goals": ["Advanced optimizer implementation", "Automated tuning", "Performance optimization"]
        },
        "phases": [
            "Advanced optimizers",
            "Hyperparameter search",
            "Learning rate scheduling",
            "Optimization strategies"
        ]
    },
    {
        "id": 10,
        "title": "CNN/LSTM Classifier Project",
        "subtitle": "Build and compare CNN and LSTM classifiers",
        "difficulty": "Advanced",
        "duration": "9-12 hours",
        "technologies": ["Classification", "Loss Analysis", "Model Comparison", "Performance"],
        "business_scenario": {
            "company": "MultiModal AI",
            "challenge": "Create a unified platform that can handle both image and sequence classification tasks",
            "goals": ["Unified classification framework", "Model comparison", "Production deployment"]
        },
        "phases": [
            "CNN classifier",
            "LSTM classifier", 
            "Performance analysis",
            "Deployment preparation"
        ]
    },
    {
        "id": 11,
        "title": "NLP Fundamentals",
        "subtitle": "Master tokenization, embeddings, and core NLP preprocessing",
        "difficulty": "Foundations",
        "duration": "5-7 hours",
        "technologies": ["NLTK", "spaCy", "Tokenization", "Word2Vec"],
        "business_scenario": {
            "company": "TextFlow Solutions",
            "challenge": "Build a comprehensive NLP preprocessing pipeline for multilingual text analysis",
            "goals": ["Robust text preprocessing", "Language-agnostic pipeline", "Scalable architecture"]
        },
        "phases": [
            "Text preprocessing",
            "Tokenization methods",
            "Word embeddings",
            "Pipeline integration"
        ]
    },
    {
        "id": 12,
        "title": "Sentiment Analysis Lab", 
        "subtitle": "Build end-to-end sentiment analysis system",
        "difficulty": "Intermediate",
        "duration": "6-8 hours",
        "technologies": ["Sentiment Analysis", "Text Classification", "Feature Engineering", "Model Evaluation"],
        "business_scenario": {
            "company": "SocialInsight Analytics",
            "challenge": "Create real-time sentiment analysis for social media monitoring and brand management",
            "goals": ["Real-time processing", "High accuracy", "Scalable deployment"]
        },
        "phases": [
            "Data collection",
            "Feature engineering",
            "Model training",
            "Evaluation metrics"
        ]
    },
    {
        "id": 13,
        "title": "Transformer Architecture",
        "subtitle": "Deep dive into BERT, GPT, and self-attention mechanisms", 
        "difficulty": "Advanced",
        "duration": "8-10 hours",
        "technologies": ["Transformers", "BERT", "GPT", "Attention"],
        "business_scenario": {
            "company": "LangModel Corp",
            "challenge": "Implement state-of-the-art transformer models for various NLP tasks",
            "goals": ["Transformer implementation", "Attention mechanisms", "Pre-trained models"]
        },
        "phases": [
            "Attention mechanisms",
            "Transformer architecture",
            "BERT implementation",
            "GPT exploration"
        ]
    },
    {
        "id": 14,
        "title": "Fine-tuning BERT/GPT",
        "subtitle": "Master fine-tuning pre-trained language models",
        "difficulty": "Advanced", 
        "duration": "7-9 hours",
        "technologies": ["HuggingFace", "Fine-tuning", "Transfer Learning", "Transformers"],
        "business_scenario": {
            "company": "LanguageTech Pro",
            "challenge": "Adapt pre-trained models for domain-specific tasks with limited data",
            "goals": ["Efficient fine-tuning", "Domain adaptation", "Performance optimization"]
        },
        "phases": [
            "Model selection",
            "Fine-tuning strategy",
            "Domain adaptation",
            "Performance evaluation"
        ]
    },
    {
        "id": 15,
        "title": "Data Engineering Pipeline",
        "subtitle": "Build robust data pipelines with web scraping and APIs",
        "difficulty": "Intermediate",
        "duration": "6-8 hours", 
        "technologies": ["Web Scraping", "APIs", "Data Pipeline", "ETL"],
        "business_scenario": {
            "company": "DataStream Systems",
            "challenge": "Create automated data collection and processing pipelines for ML workflows",
            "goals": ["Automated data collection", "Robust ETL", "Scalable architecture"]
        },
        "phases": [
            "Web scraping",
            "API integration",
            "Data processing",
            "Pipeline automation"
        ]
    },
    {
        "id": 16,
        "title": "Real-world Log Processing",
        "subtitle": "Clean and process messy real-world log data",
        "difficulty": "Intermediate",
        "duration": "5-7 hours",
        "technologies": ["Log Processing", "Data Cleaning", "Regex", "Text Processing"],
        "business_scenario": {
            "company": "LogAnalytics Plus",
            "challenge": "Process and analyze large-scale system logs for anomaly detection",
            "goals": ["Automated log processing", "Pattern recognition", "Anomaly detection"]
        },
        "phases": [
            "Log parsing",
            "Data cleaning",
            "Pattern extraction",
            "Anomaly detection"
        ]
    },
    {
        "id": 17,
        "title": "LLM Optimization Techniques", 
        "subtitle": "Implement quantization, PEFT, and knowledge distillation",
        "difficulty": "Advanced",
        "duration": "8-10 hours",
        "technologies": ["Quantization", "PEFT", "Knowledge Distillation", "LLM"],
        "business_scenario": {
            "company": "EfficientAI Labs",
            "challenge": "Optimize large language models for edge deployment and reduced computational costs",
            "goals": ["Model compression", "Efficiency optimization", "Edge deployment"]
        },
        "phases": [
            "Model quantization",
            "PEFT implementation", 
            "Knowledge distillation",
            "Performance evaluation"
        ]
    },
    {
        "id": 18,
        "title": "IT Ticket Classification Mini-Project",
        "subtitle": "Compress and fine-tune LLM for IT ticket classification",
        "difficulty": "Advanced",
        "duration": "10-12 hours",
        "technologies": ["Classification", "Fine-tuning", "Model Compression", "IT Domain"],
        "business_scenario": {
            "company": "ServiceDesk AI",
            "challenge": "Automate IT ticket routing and prioritization using advanced NLP",
            "goals": ["Automated ticket routing", "Priority classification", "Efficiency improvement"]
        },
        "phases": [
            "Data preparation",
            "Model fine-tuning",
            "Compression techniques",
            "Deployment testing"
        ]
    },
    {
        "id": 19,
        "title": "FastAPI Model Deployment",
        "subtitle": "Deploy ML models using FastAPI with MLOps pipeline",
        "difficulty": "Intermediate", 
        "duration": "6-8 hours",
        "technologies": ["FastAPI", "MLOps", "Docker", "Monitoring"],
        "business_scenario": {
            "company": "DeployML Solutions",
            "challenge": "Create production-ready ML model serving infrastructure",
            "goals": ["Scalable deployment", "Model monitoring", "CI/CD integration"]
        },
        "phases": [
            "API development",
            "Model serving",
            "Monitoring setup",
            "Production deployment"
        ]
    },
    {
        "id": 20,
        "title": "Chat Assistant Capstone (Day 1)",
        "subtitle": "Design and implement a custom chat assistant",
        "difficulty": "Capstone",
        "duration": "8-10 hours",
        "technologies": ["Chatbot", "Conversation AI", "Context Management", "NLU"],
        "business_scenario": {
            "company": "ConversationAI Corp",
            "challenge": "Build an intelligent chat assistant for customer service automation",
            "goals": ["Natural conversation", "Context awareness", "Multi-turn dialogue"]
        },
        "phases": [
            "Architecture design",
            "Core implementation",
            "Context management",
            "Testing framework"
        ]
    },
    {
        "id": 21,
        "title": "Chat Assistant Capstone (Day 2)",
        "subtitle": "Complete chat assistant with advanced features",
        "difficulty": "Capstone",
        "duration": "8-10 hours", 
        "technologies": ["Advanced Features", "Testing", "Deployment", "Production"],
        "business_scenario": {
            "company": "ConversationAI Corp",
            "challenge": "Finalize and deploy the chat assistant with production-ready features",
            "goals": ["Feature completion", "Production deployment", "Performance optimization"]
        },
        "phases": [
            "Advanced features",
            "Integration testing",
            "Performance optimization",
            "Production deployment"
        ]
    },
    {
        "id": 22,
        "title": "Azure OpenAI Integration",
        "subtitle": "Integrate Azure OpenAI and Cognitive Services",
        "difficulty": "Advanced",
        "duration": "6-8 hours",
        "technologies": ["Azure OpenAI", "Cognitive Services", "Cloud Integration", "Enterprise AI"],
        "business_scenario": {
            "company": "CloudAI Enterprise",
            "challenge": "Integrate enterprise-grade AI services for large-scale applications",
            "goals": ["Cloud integration", "Enterprise security", "Scalable architecture"]
        },
        "phases": [
            "Azure setup",
            "Service integration",
            "Security implementation", 
            "Scalability testing"
        ]
    },
    {
        "id": 23,
        "title": "LangChain Agent Development",
        "subtitle": "Build intelligent agents with LangChain",
        "difficulty": "Advanced",
        "duration": "7-9 hours",
        "technologies": ["LangChain", "Agents", "Tools", "Memory"],
        "business_scenario": {
            "company": "AgentAI Systems",
            "challenge": "Create autonomous AI agents for complex task automation",
            "goals": ["Agent intelligence", "Tool integration", "Autonomous operation"]
        },
        "phases": [
            "Agent architecture",
            "Tool integration",
            "Memory systems",
            "Multi-agent coordination"
        ]
    },
    {
        "id": 24,
        "title": "RAG System Implementation", 
        "subtitle": "Build Retrieval-Augmented Generation system",
        "difficulty": "Advanced",
        "duration": "8-10 hours",
        "technologies": ["RAG", "Vector DB", "Embeddings", "Retrieval"],
        "business_scenario": {
            "company": "KnowledgeAI Corp",
            "challenge": "Create intelligent knowledge retrieval system for enterprise documentation",
            "goals": ["Accurate retrieval", "Contextual generation", "Scalable knowledge base"]
        },
        "phases": [
            "Vector database setup",
            "Retrieval system",
            "Generation pipeline",
            "End-to-end integration"
        ]
    },
    {
        "id": 25,
        "title": "MCP Pipeline Capstone",
        "subtitle": "Build complete Model Context Protocol pipeline",
        "difficulty": "Capstone", 
        "duration": "12-15 hours",
        "technologies": ["MCP", "Multi-Agent", "Containerization", "End-to-End"],
        "business_scenario": {
            "company": "NextGen AI Platform",
            "challenge": "Create cutting-edge AI platform with Model Context Protocol integration",
            "goals": ["Advanced AI platform", "Multi-agent systems", "Production deployment"]
        },
        "phases": [
            "MCP implementation",
            "Multi-agent workflow",
            "Containerization",
            "Platform integration"
        ]
    }
]

def generate_assignment_html(assignment):
    """Generate HTML content for a single assignment"""
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Assignment {assignment['id']}: {assignment['title']} - ML Specialization</title>
    
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea, #764ba2);
            min-height: 100vh;
            color: #333;
            line-height: 1.6;
        }}

        .header {{
            background: rgba(255, 255, 255, 0.95);
            padding: 40px 0;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            margin-bottom: 40px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }}

        h1 {{
            color: #667eea;
            font-size: 2.8em;
            margin-bottom: 15px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}

        .assignment-meta {{
            background: linear-gradient(135deg, #ff6b6b, #ffd93d);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }}

        .nav-tabs {{
            background: white;
            border-radius: 15px;
            padding: 10px;
            margin-bottom: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }}

        .nav-tab {{
            padding: 12px 20px;
            background: #f8f9fa;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 500;
            color: #666;
            flex: 1;
            min-width: 140px;
            text-align: center;
        }}

        .nav-tab.active {{
            background: #667eea;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }}

        .tab-content {{
            display: none;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            margin-bottom: 30px;
        }}

        .tab-content.active {{
            display: block;
        }}

        .section-header {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 3px solid #667eea;
        }}

        .section-icon {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5em;
        }}

        .objective-card {{
            background: linear-gradient(135deg, #f8f9fa, #ffffff);
            border: 2px solid #e9ecef;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            transition: all 0.3s ease;
        }}

        .objective-card:hover {{
            border-color: #667eea;
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.15);
        }}

        .task-list {{
            background: #f8f9fa;
            border-radius: 12px;
            padding: 25px;
            margin: 20px 0;
        }}

        .task-item {{
            display: flex;
            align-items: flex-start;
            gap: 15px;
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid #e9ecef;
        }}

        .task-item:last-child {{
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }}

        .task-number {{
            background: #667eea;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            flex-shrink: 0;
        }}

        .code-block {{
            background: #2d3748;
            color: #e2e8f0;
            padding: 25px;
            border-radius: 12px;
            margin: 20px 0;
            overflow-x: auto;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            position: relative;
            border-left: 5px solid #667eea;
        }}

        .copy-btn {{
            position: absolute;
            top: 15px;
            right: 15px;
            background: #667eea;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.8em;
            transition: background 0.3s ease;
        }}

        .copy-btn:hover {{
            background: #5a6fd8;
        }}

        .warning-box {{
            background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            border: 1px solid #ffeaa7;
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .info-box {{
            background: linear-gradient(135deg, #d1ecf1, #bee5eb);
            border: 1px solid #bee5eb;
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .btn {{
            background: #667eea;
            color: white;
            padding: 12px 25px;
            text-decoration: none;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
            font-size: 1em;
        }}

        .btn:hover {{
            background: #5a6fd8;
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        }}

        .deliverable-item {{
            background: linear-gradient(135deg, #e8f5e8, #f0fff0);
            border: 1px solid #d4edda;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .deliverable-icon {{
            background: #28a745;
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2em;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 0 15px;
            }}
            
            h1 {{
                font-size: 2.2em;
            }}
            
            .nav-tabs {{
                flex-direction: column;
            }}
            
            .nav-tab {{
                min-width: auto;
            }}
            
            .assignment-meta {{
                flex-direction: column;
                text-align: center;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>📚 Assignment {assignment['id']}: {assignment['title']}</h1>
            <p>{assignment['subtitle']}</p>
        </div>
    </div>

    <div class="container">
        <div class="assignment-meta">
            <div>
                <strong>🎯 Difficulty:</strong> {assignment['difficulty']}
            </div>
            <div>
                <strong>⏱️ Duration:</strong> {assignment['duration']}
            </div>
            <div>
                <strong>🛠️ Technologies:</strong> {', '.join(assignment['technologies'])}
            </div>
            <div>
                <strong>📊 Assignment:</strong> {assignment['id']}/25
            </div>
        </div>

        <div class="nav-tabs">
            <button class="nav-tab active" onclick="showTab('overview')">📋 Overview</button>
            <button class="nav-tab" onclick="showTab('setup')">⚙️ Setup</button>
            <button class="nav-tab" onclick="showTab('tasks')">📝 Tasks</button>
            <button class="nav-tab" onclick="showTab('code')">💻 Implementation</button>
            <button class="nav-tab" onclick="showTab('evaluation')">📊 Evaluation</button>
            <button class="nav-tab" onclick="showTab('deliverables')">📦 Deliverables</button>
        </div>

        <!-- Overview Tab -->
        <div id="overview" class="tab-content active">
            <div class="section-header">
                <div class="section-icon">📋</div>
                <h2>Assignment Overview</h2>
            </div>

            <div class="objective-card">
                <h3>🎯 Learning Objectives</h3>
                <ul style="margin-top: 15px; margin-left: 20px;">
                    <li>Master the core concepts and techniques for this assignment</li>
                    <li>Implement practical solutions to real-world problems</li>
                    <li>Develop hands-on experience with industry-standard tools</li>
                    <li>Build production-ready implementations</li>
                    <li>Analyze and evaluate solution performance</li>
                </ul>
            </div>

            <div class="objective-card">
                <h3>🏢 Business Scenario: {assignment['business_scenario']['company']}</h3>
                <p><strong>Challenge:</strong> {assignment['business_scenario']['challenge']}</p>
                
                <h4 style="margin-top: 20px;">🎯 Project Goals:</h4>
                <ul style="margin-top: 10px; margin-left: 20px;">"""
    
    for goal in assignment['business_scenario']['goals']:
        html_content += f"\n                    <li>{goal}</li>"
    
    html_content += f"""
                </ul>
            </div>

            <div class="warning-box">
                <span style="font-size: 1.5em;">⚠️</span>
                <div>
                    <strong>Important:</strong> This assignment builds upon previous concepts and requires a solid understanding of the fundamentals. Make sure to complete prerequisites before starting.
                </div>
            </div>
        </div>

        <!-- Setup Tab -->
        <div id="setup" class="tab-content">
            <div class="section-header">
                <div class="section-icon">⚙️</div>
                <h2>Environment Setup</h2>
            </div>

            <div class="task-list">
                <h3>📦 Required Libraries</h3>
                <div class="code-block">
                    <button class="copy-btn" onclick="copyToClipboard(this)">Copy</button>
                    <pre>
# Core libraries for Assignment {assignment['id']}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assignment-specific libraries
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Additional tools based on assignment requirements
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
</pre>
                </div>
            </div>

            <div class="info-box">
                <span style="font-size: 1.5em;">💡</span>
                <div>
                    <strong>Setup Note:</strong> Ensure all required libraries are installed and your environment is properly configured before starting the assignment tasks.
                </div>
            </div>
        </div>

        <!-- Tasks Tab -->
        <div id="tasks" class="tab-content">
            <div class="section-header">
                <div class="section-icon">📝</div>
                <h2>Assignment Tasks</h2>
            </div>"""

    # Add phases as task sections
    for i, phase in enumerate(assignment['phases'], 1):
        html_content += f"""
            <div class="task-list">
                <h3>Phase {i}: {phase.title()}</h3>
                
                <div class="task-item">
                    <div class="task-number">{i}</div>
                    <div>
                        <h4>{phase.title()} Implementation</h4>
                        <p>Complete the implementation and analysis for {phase.lower()}. Focus on best practices and thorough documentation.</p>
                        <strong>Deliverable:</strong> Working implementation with analysis and documentation
                    </div>
                </div>
            </div>"""

    html_content += f"""
        </div>

        <!-- Code Implementation Tab -->
        <div id="code" class="tab-content">
            <div class="section-header">
                <div class="section-icon">💻</div>
                <h2>Code Implementation</h2>
            </div>

            <div class="task-list">
                <h3>🚀 Getting Started</h3>
                <div class="code-block">
                    <button class="copy-btn" onclick="copyToClipboard(this)">Copy</button>
                    <pre>
# Assignment {assignment['id']}: {assignment['title']}
# Implementation template

print("Assignment {assignment['id']}: {assignment['title']}")
print("Technologies: {', '.join(assignment['technologies'])}")

# TODO: Implement your solution here
# Follow the phases outlined in the tasks section
# Remember to document your code thoroughly

class Assignment{assignment['id']}Solution:
    def __init__(self):
        self.name = "{assignment['title']}"
        self.phase = 1
    
    def phase_1(self):
        \"\"\"Implement Phase 1: {assignment['phases'][0] if assignment['phases'] else 'Initial Implementation'}\"\"\"
        pass
    
    def phase_2(self):
        \"\"\"Implement Phase 2: {assignment['phases'][1] if len(assignment['phases']) > 1 else 'Analysis and Evaluation'}\"\"\"
        pass
    
    def evaluate_solution(self):
        \"\"\"Evaluate the complete solution\"\"\"
        pass

# Initialize solution
solution = Assignment{assignment['id']}Solution()
print(f"Ready to implement: {{solution.name}}")
</pre>
                </div>
            </div>
        </div>

        <!-- Evaluation Tab -->
        <div id="evaluation" class="tab-content">
            <div class="section-header">
                <div class="section-icon">📊</div>
                <h2>Evaluation Criteria</h2>
            </div>

            <div class="task-list">
                <h3>🎯 Technical Implementation (50 points)</h3>
                
                <div class="task-item">
                    <div class="task-number">1</div>
                    <div>
                        <h4>Core Implementation (25 points)</h4>
                        <ul>
                            <li>Correct implementation of key algorithms and techniques</li>
                            <li>Proper use of libraries and frameworks</li>
                            <li>Code quality and best practices</li>
                        </ul>
                    </div>
                </div>

                <div class="task-item">
                    <div class="task-number">2</div>
                    <div>
                        <h4>Performance and Optimization (25 points)</h4>
                        <ul>
                            <li>Solution efficiency and optimization</li>
                            <li>Proper evaluation metrics and analysis</li>
                            <li>Comparison with baseline approaches</li>
                        </ul>
                    </div>
                </div>
            </div>

            <div class="task-list">
                <h3>📈 Analysis and Documentation (35 points)</h3>
                
                <div class="task-item">
                    <div class="task-number">3</div>
                    <div>
                        <h4>Comprehensive Analysis (20 points)</h4>
                        <ul>
                            <li>Thorough analysis of results and findings</li>
                            <li>Clear explanation of methodology and approach</li>
                            <li>Insights and recommendations</li>
                        </ul>
                    </div>
                </div>

                <div class="task-item">
                    <div class="task-number">4</div>
                    <div>
                        <h4>Documentation Quality (15 points)</h4>
                        <ul>
                            <li>Clear and comprehensive documentation</li>
                            <li>Professional presentation of results</li>
                            <li>Code comments and explanations</li>
                        </ul>
                    </div>
                </div>
            </div>

            <div class="task-list">
                <h3>💼 Business Application (15 points)</h3>
                
                <div class="task-item">
                    <div class="task-number">5</div>
                    <div>
                        <h4>Real-world Application (15 points)</h4>
                        <ul>
                            <li>Practical application to business scenario</li>
                            <li>Actionable insights and recommendations</li>
                            <li>Consideration of deployment and scalability</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <!-- Deliverables Tab -->
        <div id="deliverables" class="tab-content">
            <div class="section-header">
                <div class="section-icon">📦</div>
                <h2>Project Deliverables</h2>
            </div>

            <div class="deliverable-item">
                <div class="deliverable-icon">📊</div>
                <div>
                    <h4>1. Complete Implementation</h4>
                    <p>Fully working implementation of all assignment requirements with comprehensive testing.</p>
                    <strong>Format:</strong> Jupyter notebook with detailed explanations
                </div>
            </div>

            <div class="deliverable-item">
                <div class="deliverable-icon">📈</div>
                <div>
                    <h4>2. Analysis Report</h4>
                    <p>Comprehensive analysis of results, methodology, and performance evaluation.</p>
                    <strong>Format:</strong> Technical report with visualizations and insights
                </div>
            </div>

            <div class="deliverable-item">
                <div class="deliverable-icon">🔬</div>
                <div>
                    <h4>3. Business Application</h4>
                    <p>Application of solution to the business scenario with actionable recommendations.</p>
                    <strong>Format:</strong> Executive summary with business metrics and recommendations
                </div>
            </div>

            <div class="deliverable-item">
                <div class="deliverable-icon">💻</div>
                <div>
                    <h4>4. Source Code</h4>
                    <p>Clean, well-documented source code with proper structure and comments.</p>
                    <strong>Format:</strong> Python files or notebook with clear organization
                </div>
            </div>

            <div style="text-align: center; margin-top: 40px;">"""
    
    # Navigation links
    prev_id = assignment['id'] - 1 if assignment['id'] > 1 else 25
    next_id = assignment['id'] + 1 if assignment['id'] < 25 else 1
    
    html_content += f"""
                <a href="ml_practical_assignment_{prev_id}.html" class="btn">
                    ← Previous Assignment
                </a>
                <a href="ml_specialization_assignments_index.html" class="btn" style="margin: 0 15px;">
                    Back to All Assignments
                </a>
                <a href="ml_practical_assignment_{next_id}.html" class="btn">
                    Next Assignment →
                </a>
            </div>
        </div>
    </div>

    <script>
        function showTab(tabName) {{
            // Hide all tab contents
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(tab => tab.classList.remove('active'));
            
            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.nav-tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab and mark as active
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }}

        function copyToClipboard(button) {{
            const codeBlock = button.parentNode;
            const code = codeBlock.querySelector('pre').textContent;
            
            navigator.clipboard.writeText(code).then(() => {{
                const originalText = button.textContent;
                button.textContent = 'Copied!';
                button.style.background = '#28a745';
                
                setTimeout(() => {{
                    button.textContent = originalText;
                    button.style.background = '#667eea';
                }}, 2000);
            }});
        }}
    </script>
</body>
</html>"""

    return html_content

def create_all_assignments():
    """Create all 25 assignments"""
    base_path = "/Users/niranjan/Downloads/specialization_track"
    
    print("Creating 25 ML Specialization Assignments...")
    
    for assignment in ASSIGNMENTS_CONFIG:
        filename = f"ml_practical_assignment_{assignment['id']}.html"
        filepath = os.path.join(base_path, filename)
        
        html_content = generate_assignment_html(assignment)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Created Assignment {assignment['id']}: {assignment['title']}")
    
    print(f"\n🎉 Successfully created all 25 assignments!")
    print(f"📁 Files saved to: {base_path}")
    print("\n📝 Assignment Summary:")
    for assignment in ASSIGNMENTS_CONFIG:
        print(f"  {assignment['id']:2}. {assignment['title']} ({assignment['difficulty']}, {assignment['duration']})")

if __name__ == "__main__":
    create_all_assignments()