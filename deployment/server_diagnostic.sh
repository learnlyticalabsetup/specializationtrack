#!/bin/bash

# Diagnostic script for assets.learnlytica.us deployment
# Run this on your server to check the deployment

echo "🔍 Cloud Specialization Track - Deployment Diagnostic"
echo "=================================================="

# Find the web root where courses directory exists
WEB_ROOT=""
if [ -d "/var/www/html/courses" ]; then
    WEB_ROOT="/var/www/html"
elif [ -d "/usr/share/nginx/html/courses" ]; then
    WEB_ROOT="/usr/share/nginx/html"
elif [ -d "/home/*/public_html/courses" ]; then
    WEB_ROOT=$(find /home -name "courses" -type d 2>/dev/null | head -1 | xargs dirname)
else
    echo "❌ Cannot find 'courses' directory. Please specify the path:"
    read -p "Enter the full path where 'courses' directory is located: " WEB_ROOT
fi

echo "📁 Web root detected: $WEB_ROOT"

# Check directory structure
echo ""
echo "📂 Directory structure:"
ls -la "$WEB_ROOT" | grep -E "(courses|qreview)"

# Check if qreview exists
if [ -d "$WEB_ROOT/qreview" ]; then
    echo ""
    echo "📂 qreview directory contents:"
    ls -la "$WEB_ROOT/qreview"
    
    # Check both possible directory names
    if [ -d "$WEB_ROOT/qreview/speializationtrack" ]; then
        echo "⚠️  Found 'speializationtrack' (with typo)"
        SPEC_DIR="$WEB_ROOT/qreview/speializationtrack"
    elif [ -d "$WEB_ROOT/qreview/specializationtrack" ]; then
        echo "✅ Found 'specializationtrack' (correct spelling)"
        SPEC_DIR="$WEB_ROOT/qreview/specializationtrack"
    else
        echo "❌ No specialization track directory found"
        exit 1
    fi
    
    echo ""
    echo "📂 Specialization track directory contents:"
    ls -la "$SPEC_DIR" | head -10
    
    echo ""
    echo "🔍 Checking index.html:"
    if [ -f "$SPEC_DIR/index.html" ]; then
        echo "✅ index.html exists"
        echo "📄 First few lines of index.html:"
        head -5 "$SPEC_DIR/index.html"
        
        echo ""
        echo "🔍 Checking for external resource references:"
        if grep -q "main\." "$SPEC_DIR/index.html"; then
            echo "⚠️  Found references to main.css or main.js files"
            grep "main\." "$SPEC_DIR/index.html"
        else
            echo "✅ No problematic main.css/main.js references found"
        fi
        
        if grep -q "SourceSans\|SourceSerif" "$SPEC_DIR/index.html"; then
            echo "⚠️  Found references to Source font files"
            grep "SourceSans\|SourceSerif" "$SPEC_DIR/index.html"
        else
            echo "✅ No problematic font references found"
        fi
        
    else
        echo "❌ index.html does not exist!"
    fi
    
    echo ""
    echo "🔍 File permissions:"
    ls -la "$SPEC_DIR/index.html"
    
    echo ""
    echo "🔍 Ownership comparison with working courses directory:"
    echo "courses/genai ownership:"
    ls -ld "$WEB_ROOT/courses/genai" 2>/dev/null || echo "genai directory not found"
    echo "qreview/specializationtrack ownership:"
    ls -ld "$SPEC_DIR"
    
else
    echo "❌ qreview directory not found in $WEB_ROOT"
fi

echo ""
echo "🌐 Expected URL: https://assets.learnlytica.us/qreview/specializationtrack/"
echo ""
echo "🔧 Quick fixes:"
echo "1. If directory name has typo: mv speializationtrack specializationtrack"
echo "2. Fix permissions: chmod -R 755 qreview/specializationtrack"
echo "3. Fix ownership: chown -R --reference=courses/genai qreview/specializationtrack"