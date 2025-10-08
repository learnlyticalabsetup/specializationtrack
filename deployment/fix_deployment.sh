#!/bin/bash

# Quick Fix Script for assets.learnlytica.us deployment
# Run this on your server to fix the deployment issues

echo "🔧 Fixing Cloud Specialization Track Deployment..."

# Check current directory
if [ -d "/qreview/speializationtrack" ]; then
    echo "📁 Found directory with typo, fixing..."
    mv /qreview/speializationtrack /qreview/specializationtrack
    echo "✅ Directory renamed to 'specializationtrack'"
fi

# Navigate to correct directory
cd /qreview/specializationtrack

# Check if index.html exists
if [ ! -f "index.html" ]; then
    echo "❌ index.html not found! Please copy all files properly."
    exit 1
fi

echo "✅ index.html found"

# Set proper permissions
echo "🔐 Setting file permissions..."
chmod -R 755 .
chmod -R 777 deployment/comments_tracking

# Detect web server user and set ownership
if id "apache" &>/dev/null; then
    chown -R apache:apache .
    echo "✅ Set ownership to apache:apache"
elif id "nginx" &>/dev/null; then
    chown -R nginx:nginx .
    echo "✅ Set ownership to nginx:nginx"
elif id "www-data" &>/dev/null; then
    chown -R www-data:www-data .
    echo "✅ Set ownership to www-data:www-data"
else
    echo "⚠️  Could not detect web server user, keeping current ownership"
fi

# Create .htaccess for Apache (if needed)
if command -v httpd &> /dev/null || command -v apache2 &> /dev/null; then
    echo "🌐 Creating .htaccess for Apache..."
    cat > .htaccess << 'EOF'
DirectoryIndex index.html
Options +Indexes

<Files "*.csv">
    Header set Content-Type "text/csv"
    Header set Content-Disposition "attachment"
</Files>

<Directory "deployment/comments_tracking">
    Options +Indexes
    IndexOptions FancyIndexing
</Directory>
EOF
    echo "✅ .htaccess created"
fi

# Test file accessibility
echo "🔍 Testing file accessibility..."
if [ -r "index.html" ]; then
    echo "✅ index.html is readable"
else
    echo "❌ index.html permission issue"
fi

if [ -r "azure_solution_guide.html" ]; then
    echo "✅ azure_solution_guide.html is readable"
else
    echo "❌ azure_solution_guide.html permission issue"
fi

if [ -r "deployment/review_dashboard.html" ]; then
    echo "✅ review_dashboard.html is readable"
else
    echo "❌ review_dashboard.html permission issue"
fi

echo ""
echo "🎉 DEPLOYMENT FIX COMPLETE!"
echo "=========================="
echo "🌐 Your site should now be accessible at:"
echo "   https://assets.learnlytica.us/qreview/specializationtrack/"
echo ""
echo "📊 Dashboard URL:"
echo "   https://assets.learnlytica.us/qreview/specializationtrack/deployment/review_dashboard.html"
echo ""
echo "🔧 Azure Guide URL:"
echo "   https://assets.learnlytica.us/qreview/specializationtrack/azure_solution_guide.html"
echo ""
echo "🔍 If still not working, check:"
echo "   - Web server error logs"
echo "   - SELinux settings (if enabled)"
echo "   - Firewall rules"
echo ""

# Show file structure
echo "📁 Current file structure:"
ls -la | head -10