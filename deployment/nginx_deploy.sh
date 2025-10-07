#!/bin/bash

# Nginx Deployment Script for Cloud Specialization Track
# Simple copy and configure script for existing nginx installations

echo "🚀 Deploying Cloud Specialization Track to Nginx..."

# Configuration - Modify these paths as needed
NGINX_WEB_ROOT="/var/www/html"
SITE_NAME="specialization_track"
CURRENT_DIR=$(pwd)
SOURCE_DIR="/Users/niranjan/Downloads/specialization_track"

# Check if nginx is running
if ! systemctl is-active --quiet nginx; then
    echo "❌ Nginx is not running. Please start nginx first:"
    echo "   sudo systemctl start nginx"
    exit 1
fi

echo "✅ Nginx is running"

# Create target directory
TARGET_DIR="$NGINX_WEB_ROOT/$SITE_NAME"
echo "📁 Creating target directory: $TARGET_DIR"
sudo mkdir -p $TARGET_DIR

# Copy all files
echo "📄 Copying files from $SOURCE_DIR to $TARGET_DIR..."
sudo cp -r $SOURCE_DIR/* $TARGET_DIR/

# Set proper permissions
echo "🔐 Setting permissions..."
sudo chown -R www-data:www-data $TARGET_DIR
sudo chmod -R 755 $TARGET_DIR
sudo chmod -R 777 $TARGET_DIR/deployment/comments_tracking

# Create nginx site configuration
DOMAIN="localhost"
read -p "Enter your domain name (or press Enter for localhost): " USER_DOMAIN
if [ ! -z "$USER_DOMAIN" ]; then
    DOMAIN=$USER_DOMAIN
fi

echo "🌐 Creating nginx configuration for $DOMAIN..."
sudo tee /etc/nginx/sites-available/$SITE_NAME > /dev/null << EOF
server {
    listen 80;
    server_name $DOMAIN;
    root $TARGET_DIR;
    index index.html review_dashboard.html;

    # Enable gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    # Main location
    location / {
        try_files \$uri \$uri/ =404;
    }

    # CSV file downloads
    location ~* \.csv\$ {
        add_header Content-Type "text/csv";
        add_header Content-Disposition "attachment";
    }

    # Comments tracking directory
    location /deployment/comments_tracking/ {
        autoindex on;
        autoindex_exact_size off;
        autoindex_localtime on;
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;

    # Logs
    access_log /var/log/nginx/specialization_track_access.log;
    error_log /var/log/nginx/specialization_track_error.log;
}
EOF

# Enable the site
echo "⚙️ Enabling nginx site..."
sudo ln -sf /etc/nginx/sites-available/$SITE_NAME /etc/nginx/sites-enabled/

# Test nginx configuration
echo "🔍 Testing nginx configuration..."
if sudo nginx -t; then
    echo "✅ Nginx configuration is valid"
    
    # Reload nginx
    echo "🔄 Reloading nginx..."
    sudo systemctl reload nginx
    
    echo ""
    echo "🎉 DEPLOYMENT SUCCESSFUL!"
    echo "========================="
    echo "🌐 Website URL: http://$DOMAIN/$SITE_NAME/"
    echo "📊 Dashboard: http://$DOMAIN/$SITE_NAME/review_dashboard.html"
    echo "🔧 Azure Guide: http://$DOMAIN/$SITE_NAME/azure_solution_guide.html"
    echo "📁 Comments: http://$DOMAIN/$SITE_NAME/deployment/comments_tracking/"
    echo ""
    echo "📝 Files location: $TARGET_DIR"
    echo "⚙️ Nginx config: /etc/nginx/sites-available/$SITE_NAME"
    echo "📊 Logs: /var/log/nginx/specialization_track_*.log"
    echo ""
    echo "🔧 Useful commands:"
    echo "   sudo systemctl status nginx"
    echo "   sudo systemctl reload nginx"
    echo "   sudo tail -f /var/log/nginx/specialization_track_access.log"
    
else
    echo "❌ Nginx configuration test failed!"
    echo "Please check the configuration and try again."
    exit 1
fi

# Create a simple monitoring script for nginx
echo "📊 Creating monitoring script..."
sudo tee $TARGET_DIR/deployment/nginx_monitor.sh > /dev/null << 'EOF'
#!/bin/bash

# Simple monitoring script for nginx deployment
LOG_FILE="/var/log/nginx/specialization_track_access.log"
COMMENTS_DIR="/var/www/html/specialization_track/deployment/comments_tracking"

echo "=== Nginx Specialization Track Status ==="
echo "Date: $(date)"
echo ""

# Check nginx status
echo "Nginx Status:"
systemctl is-active nginx && echo "✅ Running" || echo "❌ Not running"
echo ""

# Check recent access
echo "Recent Access (last 10 requests):"
if [ -f "$LOG_FILE" ]; then
    tail -10 "$LOG_FILE" | awk '{print $1, $4, $7}' || echo "No recent access"
else
    echo "Log file not found"
fi
echo ""

# Check comment files
echo "Comment Files Status:"
if [ -d "$COMMENTS_DIR" ]; then
    ls -la "$COMMENTS_DIR"/*.csv 2>/dev/null | wc -l | xargs echo "CSV files:"
    find "$COMMENTS_DIR" -name "*.csv" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print "Total comments:", $1}'
else
    echo "Comments directory not found"
fi
echo ""

# Check disk usage
echo "Disk Usage:"
df -h /var/www/html | tail -1 | awk '{print "Available space:", $4, "(" $5 " used)"}'
EOF

chmod +x $TARGET_DIR/deployment/nginx_monitor.sh

echo ""
echo "🔍 To monitor your deployment, run:"
echo "   sudo $TARGET_DIR/deployment/nginx_monitor.sh"
echo ""
echo "✨ Your Cloud Specialization Track is now live!"