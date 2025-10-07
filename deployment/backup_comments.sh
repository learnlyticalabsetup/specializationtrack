#!/bin/bash

# Comment Backup and Maintenance Script
# Automatically backs up CSV files and generates reports

COMMENTS_DIR="/var/www/html/specialization_track/deployment/comments_tracking"
BACKUP_DIR="/var/backups/specialization_track/comments"
REPORT_DIR="/var/www/html/specialization_track/deployment/reports"
LOG_FILE="/var/log/specialization_maintenance.log"

# Create directories
mkdir -p $BACKUP_DIR
mkdir -p $REPORT_DIR

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

log_message "Starting comment backup and maintenance..."

# Backup CSV files
BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
cp $COMMENTS_DIR/aws_comments.csv $BACKUP_DIR/aws_comments_$BACKUP_TIMESTAMP.csv
cp $COMMENTS_DIR/azure_comments.csv $BACKUP_DIR/azure_comments_$BACKUP_TIMESTAMP.csv
cp $COMMENTS_DIR/review_summary.csv $BACKUP_DIR/review_summary_$BACKUP_TIMESTAMP.csv

log_message "Backup completed: $BACKUP_TIMESTAMP"

# Generate daily report
python3 << EOF
import csv
import json
from datetime import datetime, timedelta

# Read AWS comments
aws_comments = []
with open('$COMMENTS_DIR/aws_comments.csv', 'r') as f:
    reader = csv.DictReader(f)
    aws_comments = list(reader)

# Read Azure comments
azure_comments = []
with open('$COMMENTS_DIR/azure_comments.csv', 'r') as f:
    reader = csv.DictReader(f)
    azure_comments = list(reader)

# Generate statistics
today = datetime.now().strftime('%Y-%m-%d')
aws_today = [c for c in aws_comments if c['timestamp'].startswith(today)]
azure_today = [c for c in azure_comments if c['timestamp'].startswith(today)]

stats = {
    'date': today,
    'aws_total': len(aws_comments),
    'azure_total': len(azure_comments),
    'aws_today': len(aws_today),
    'azure_today': len(azure_today),
    'aws_action_items': len([c for c in aws_comments if c['action_required'].lower() == 'yes']),
    'azure_action_items': len([c for c in azure_comments if c['action_required'].lower() == 'yes']),
    'aws_avg_rating': sum(float(c['rating']) for c in aws_comments if c['rating']) / len(aws_comments) if aws_comments else 0,
    'azure_avg_rating': sum(float(c['rating']) for c in azure_comments if c['rating']) / len(azure_comments) if azure_comments else 0
}

# Save daily report
with open('$REPORT_DIR/daily_report_$BACKUP_TIMESTAMP.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"Daily report generated: {stats}")
EOF

# Cleanup old backups (keep last 30 days)
find $BACKUP_DIR -name "*.csv" -mtime +30 -delete
find $REPORT_DIR -name "*.json" -mtime +30 -delete

log_message "Maintenance completed. Old files cleaned up."

# Check disk space
DISK_USAGE=$(df $COMMENTS_DIR | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
    log_message "WARNING: Disk usage is at $DISK_USAGE%"
fi

log_message "Backup and maintenance script completed."