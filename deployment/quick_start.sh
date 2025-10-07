#!/bin/bash

# Quick Start Script for Cloud Specialization Track
# Run this script to set up the environment quickly

echo "🚀 Cloud Specialization Track - Quick Start"
echo "==========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "⚠️  Please run as root or with sudo"
    exit 1
fi

# Update system
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install required packages
echo "🔧 Installing required packages..."
apt install -y apache2 python3 python3-pip certbot python3-certbot-apache

# Enable Apache modules
echo "⚙️  Enabling Apache modules..."
a2enmod rewrite
a2enmod headers
a2enmod ssl

# Set domain (prompt user)
read -p "Enter your domain name (e.g., example.com): " DOMAIN
if [ -z "$DOMAIN" ]; then
    DOMAIN="localhost"
    echo "Using localhost for local testing"
fi

# Set paths
SERVER_PATH="/var/www/html/specialization_track"
CURRENT_DIR=$(pwd)

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p $SERVER_PATH
mkdir -p $SERVER_PATH/deployment/comments_tracking
mkdir -p $SERVER_PATH/deployment/logs
mkdir -p $SERVER_PATH/deployment/aws_reviews
mkdir -p $SERVER_PATH/deployment/azure_reviews
mkdir -p /var/backups/specialization_track

# Copy files
echo "📄 Copying files..."
cp $CURRENT_DIR/*.html $SERVER_PATH/ 2>/dev/null || echo "HTML files not found in current directory"
cp $CURRENT_DIR/deployment/*.csv $SERVER_PATH/deployment/comments_tracking/ 2>/dev/null || echo "CSV files not found"
cp $CURRENT_DIR/deployment/*.sh $SERVER_PATH/deployment/ 2>/dev/null || echo "Shell scripts not found"

# Set permissions
echo "🔐 Setting permissions..."
chown -R www-data:www-data $SERVER_PATH
chmod -R 755 $SERVER_PATH
chmod -R 777 $SERVER_PATH/deployment/comments_tracking

# Create Apache configuration
echo "🌐 Configuring Apache..."
cat > /etc/apache2/sites-available/specialization_track.conf << EOF
<VirtualHost *:80>
    ServerName $DOMAIN
    DocumentRoot $SERVER_PATH
    DirectoryIndex index.html review_dashboard.html
    
    <Directory $SERVER_PATH>
        Options Indexes FollowSymLinks
        AllowOverride All
        Require all granted
    </Directory>
    
    <Files "*.csv">
        Header set Content-Type "text/csv"
        Header set Content-Disposition "attachment"
    </Files>
    
    ErrorLog /var/log/apache2/specialization_track_error.log
    CustomLog /var/log/apache2/specialization_track_access.log combined
</VirtualHost>
EOF

# Enable site
a2ensite specialization_track.conf
a2dissite 000-default.conf
systemctl reload apache2

# Create monitoring cron job
echo "⏰ Setting up monitoring..."
(crontab -l 2>/dev/null; echo "0 */6 * * * $SERVER_PATH/deployment/backup_comments.sh") | crontab -

# Setup SSL if domain is not localhost
if [ "$DOMAIN" != "localhost" ]; then
    read -p "Enter email for SSL certificate: " EMAIL
    if [ ! -z "$EMAIL" ]; then
        echo "🔒 Setting up SSL certificate..."
        certbot --apache -d $DOMAIN --non-interactive --agree-tos --email $EMAIL
    fi
fi

# Create default index if not exists
if [ ! -f "$SERVER_PATH/index.html" ]; then
    echo "📝 Creating default index..."
    cat > $SERVER_PATH/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Cloud Specialization Track</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
        .container { max-width: 600px; margin: 0 auto; }
        .btn { display: inline-block; padding: 15px 30px; margin: 10px; 
               background: #007bff; color: white; text-decoration: none; 
               border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Cloud Specialization Track</h1>
        <p>Welcome to your cloud learning platform!</p>
        <a href="review_dashboard.html" class="btn">Review Dashboard</a>
        <a href="azure_solution_guide.html" class="btn">Azure Guide</a>
        <a href="deployment/comments_tracking/" class="btn">Comments Tracking</a>
    </div>
</body>
</html>
EOF
fi

# Final status check
echo ""
echo "✅ INSTALLATION COMPLETE!"
echo "========================="
echo "🌐 Website URL: http://$DOMAIN"
if [ "$DOMAIN" != "localhost" ]; then
    echo "🔒 HTTPS URL: https://$DOMAIN"
fi
echo "📊 Dashboard: http://$DOMAIN/review_dashboard.html"
echo "📁 Files location: $SERVER_PATH"
echo "📝 Logs: /var/log/apache2/specialization_track_*.log"
echo ""
echo "🔧 To manage the site:"
echo "   systemctl status apache2"
echo "   systemctl restart apache2"
echo "   cd $SERVER_PATH"
echo ""
echo "📊 To check comments:"
echo "   ls -la $SERVER_PATH/deployment/comments_tracking/"
echo ""

# Test the installation
if curl -s "http://localhost" > /dev/null; then
    echo "✅ Website is accessible!"
else
    echo "❌ Website test failed. Check Apache configuration."
fi

echo "🎉 Setup completed! Your Cloud Specialization Track is ready!"