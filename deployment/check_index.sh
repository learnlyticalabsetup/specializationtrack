#!/bin/bash

# Index.html diagnostic script
# Check what external references exist in the deployed index.html

echo "🔍 Index.html External References Diagnostic"
echo "============================================"

# Find the deployed index.html file
echo "📍 Looking for deployed index.html files..."

# Check common web root locations
POSSIBLE_PATHS=(
    "/var/www/html/qreview/specializationtrack/index.html"
    "/var/www/html/qreview/speializationtrack/index.html"
    "/usr/share/nginx/html/qreview/specializationtrack/index.html"
    "/usr/share/nginx/html/qreview/speializationtrack/index.html"
)

INDEX_FILE=""
for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -f "$path" ]; then
        INDEX_FILE="$path"
        echo "✅ Found: $path"
        break
    fi
done

if [ -z "$INDEX_FILE" ]; then
    echo "❌ Index.html not found in standard locations."
    echo "Please provide the full path to your deployed index.html:"
    read -p "Path: " INDEX_FILE
fi

if [ ! -f "$INDEX_FILE" ]; then
    echo "❌ File not found: $INDEX_FILE"
    exit 1
fi

echo ""
echo "🔍 Analyzing: $INDEX_FILE"
echo "----------------------------------------"

# Check file size and basic info
echo "📊 File info:"
ls -lh "$INDEX_FILE"

echo ""
echo "🔍 Searching for external references..."

# Look for CSS files
echo ""
echo "📄 CSS References:"
grep -n -i "\.css\|main\." "$INDEX_FILE" | head -10 || echo "   No CSS references found"

# Look for JS files
echo ""
echo "📄 JavaScript References:"
grep -n -i "\.js\|main\." "$INDEX_FILE" | head -10 || echo "   No JS references found"

# Look for font files
echo ""
echo "🔤 Font References:"
grep -n -i "\.woff\|\.woff2\|font\|SourceSans\|SourceSerif" "$INDEX_FILE" | head -10 || echo "   No font references found"

# Look for any src or href attributes
echo ""
echo "🔗 All External Links (src/href):"
grep -n -i "src=\|href=" "$INDEX_FILE" | head -15 || echo "   No external links found"

# Show first 20 lines of the file
echo ""
echo "📄 First 20 lines of the file:"
echo "----------------------------------------"
head -20 "$INDEX_FILE"

echo ""
echo "📄 Title and any link/script tags:"
echo "----------------------------------------"
grep -i "<title>\|<link\|<script" "$INDEX_FILE" || echo "   No title/link/script tags found"

echo ""
echo "🔍 Check for build artifacts (webpack, etc.):"
echo "----------------------------------------"
if grep -q "webpack\|build\|bundle\|chunk" "$INDEX_FILE"; then
    echo "⚠️  Found build-related content - this might be a different project!"
    grep -n "webpack\|build\|bundle\|chunk" "$INDEX_FILE"
else
    echo "✅ No build artifacts found"
fi

echo ""
echo "💡 Recommendations:"
echo "- If you see references to main.css, main.js, or SourceSans fonts,"
echo "  this is NOT the correct index.html from our project"
echo "- Our index.html should be self-contained with embedded CSS"
echo "- If wrong file is deployed, replace it with the correct one"