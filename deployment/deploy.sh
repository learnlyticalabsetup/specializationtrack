#!/bin/bash

# Cloud Specialization Track - Deployment Script
# This script automates the deployment process for both AWS and Azure training materials

echo "🚀 Starting Cloud Specialization Track Deployment..."

# Configuration
DOMAIN="your-domain.com"
SERVER_PATH="/var/www/html/specialization_track"
BACKUP_PATH="/var/backups/specialization_track"
LOG_FILE="/var/log/specialization_deployment.log"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# Create backup of existing deployment
if [ -d "$SERVER_PATH" ]; then
    log_message "Creating backup of existing deployment..."
    mkdir -p $BACKUP_PATH
    cp -r $SERVER_PATH $BACKUP_PATH/backup_$(date +%Y%m%d_%H%M%S)
fi

# Create directory structure
log_message "Creating directory structure..."
mkdir -p $SERVER_PATH
mkdir -p $SERVER_PATH/deployment/comments_tracking
mkdir -p $SERVER_PATH/deployment/logs
mkdir -p $SERVER_PATH/deployment/aws_reviews
mkdir -p $SERVER_PATH/deployment/azure_reviews

# Copy files
log_message "Copying files to server..."
cp *.html $SERVER_PATH/
cp *.csv $SERVER_PATH/
cp -r deployment/* $SERVER_PATH/deployment/

# Set permissions
log_message "Setting file permissions..."
chmod 755 $SERVER_PATH
chmod 644 $SERVER_PATH/*.html
chmod 644 $SERVER_PATH/*.csv
chmod 755 $SERVER_PATH/deployment
chmod 777 $SERVER_PATH/deployment/comments_tracking
chmod 755 $SERVER_PATH/deployment/logs

# Initialize comment tracking files if they don't exist
if [ ! -f "$SERVER_PATH/deployment/comments_tracking/aws_comments.csv" ]; then
    log_message "Initializing AWS comments file..."
    echo "timestamp,reviewer_id,assignment_type,assignment_number,section,comment_type,rating,comment_text,action_required,status,resolution_date" > $SERVER_PATH/deployment/comments_tracking/aws_comments.csv
fi

if [ ! -f "$SERVER_PATH/deployment/comments_tracking/azure_comments.csv" ]; then
    log_message "Initializing Azure comments file..."
    echo "timestamp,reviewer_id,assignment_type,assignment_number,section,comment_type,rating,comment_text,action_required,status,resolution_date" > $SERVER_PATH/deployment/comments_tracking/azure_comments.csv
fi

# Create Apache virtual host configuration
log_message "Creating Apache configuration..."
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
    
    # Enable CSV downloads
    <Files "*.csv">
        Header set Content-Type "text/csv"
        Header set Content-Disposition "attachment"
    </Files>
    
    # Logging
    ErrorLog /var/log/apache2/specialization_track_error.log
    CustomLog /var/log/apache2/specialization_track_access.log combined
</VirtualHost>
EOF

# Enable site
a2ensite specialization_track.conf
systemctl reload apache2

# Create monitoring script
log_message "Creating monitoring script..."
cat > $SERVER_PATH/deployment/monitor_comments.sh << 'EOF'
#!/bin/bash

# Comment monitoring script
COMMENTS_DIR="/var/www/html/specialization_track/deployment/comments_tracking"
LOG_FILE="/var/log/specialization_comments.log"

# Count comments in each file
AWS_COMMENTS=$(wc -l < $COMMENTS_DIR/aws_comments.csv)
AZURE_COMMENTS=$(wc -l < $COMMENTS_DIR/azure_comments.csv)

# Log statistics
echo "[$(date '+%Y-%m-%d %H:%M:%S')] AWS Comments: $AWS_COMMENTS, Azure Comments: $AZURE_COMMENTS" >> $LOG_FILE

# Check for action items
AWS_ACTIONS=$(grep -c "yes" $COMMENTS_DIR/aws_comments.csv)
AZURE_ACTIONS=$(grep -c "yes" $COMMENTS_DIR/azure_comments.csv)

if [ $AWS_ACTIONS -gt 0 ] || [ $AZURE_ACTIONS -gt 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ACTION REQUIRED: AWS($AWS_ACTIONS) Azure($AZURE_ACTIONS)" >> $LOG_FILE
fi
EOF

chmod +x $SERVER_PATH/deployment/monitor_comments.sh

# Create cron job for monitoring
log_message "Setting up monitoring cron job..."
(crontab -l 2>/dev/null; echo "0 * * * * $SERVER_PATH/deployment/monitor_comments.sh") | crontab -

# Create SSL configuration (if Let's Encrypt is available)
if command -v certbot &> /dev/null; then
    log_message "Setting up SSL certificate..."
    certbot --apache -d $DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN
fi

# Final verification
log_message "Verifying deployment..."
if [ -f "$SERVER_PATH/index.html" ] && [ -f "$SERVER_PATH/review_dashboard.html" ]; then
    log_message "✅ Deployment completed successfully!"
    log_message "📊 Review Dashboard: https://$DOMAIN/review_dashboard.html"
    log_message "🌐 Main Site: https://$DOMAIN/"
    log_message "📁 Comments Tracking: $SERVER_PATH/deployment/comments_tracking/"
else
    log_message "❌ Deployment verification failed!"
    exit 1
fi

# Display deployment summary
echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo "=========================="
echo "Main Site: https://$DOMAIN/"
echo "Review Dashboard: https://$DOMAIN/review_dashboard.html"
echo "AWS Content: https://$DOMAIN/aws_index.html"
echo "Azure Content: https://$DOMAIN/azure_index.html"
echo ""
echo "📊 Comment Tracking:"
echo "- AWS Comments: $SERVER_PATH/deployment/comments_tracking/aws_comments.csv"
echo "- Azure Comments: $SERVER_PATH/deployment/comments_tracking/azure_comments.csv"
echo "- Summary Report: $SERVER_PATH/deployment/comments_tracking/review_summary.csv"
echo ""
echo "📝 Logs: $LOG_FILE"
echo "🔧 Monitor: $SERVER_PATH/deployment/monitor_comments.sh"