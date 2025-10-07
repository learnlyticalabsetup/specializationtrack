# Nginx Deployment Guide
## Cloud Specialization Track

### Quick Start (Recommended)

Since you already have nginx running, here's the simplest approach:

```bash
# 1. Navigate to your project directory
cd /Users/niranjan/Downloads/specialization_track

# 2. Run the nginx deployment script
sudo ./deployment/nginx_deploy.sh
```

### Manual Copy Method

If you prefer to copy manually:

```bash
# 1. Copy the entire folder to nginx web root
sudo cp -r /Users/niranjan/Downloads/specialization_track /var/www/html/

# 2. Set permissions
sudo chown -R www-data:www-data /var/www/html/specialization_track
sudo chmod -R 755 /var/www/html/specialization_track
sudo chmod -R 777 /var/www/html/specialization_track/deployment/comments_tracking

# 3. Create nginx site configuration
sudo nano /etc/nginx/sites-available/specialization_track
```

### Nginx Configuration Template

Add this to `/etc/nginx/sites-available/specialization_track`:

```nginx
server {
    listen 80;
    server_name your-domain.com;  # Change this to your domain or localhost
    root /var/www/html/specialization_track;
    index index.html review_dashboard.html;

    location / {
        try_files $uri $uri/ =404;
    }

    # CSV downloads
    location ~* \.csv$ {
        add_header Content-Type "text/csv";
        add_header Content-Disposition "attachment";
    }

    # Comments tracking
    location /deployment/comments_tracking/ {
        autoindex on;
    }

    access_log /var/log/nginx/specialization_track_access.log;
    error_log /var/log/nginx/specialization_track_error.log;
}
```

### Enable the Site

```bash
# Enable the site
sudo ln -s /etc/nginx/sites-available/specialization_track /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

### Access Your Site

After deployment, access your site at:
- **Main Portal**: `http://your-domain.com/specialization_track/`
- **Review Dashboard**: `http://your-domain.com/specialization_track/review_dashboard.html`
- **Azure Guide**: `http://your-domain.com/specialization_track/azure_solution_guide.html`
- **Comments Tracking**: `http://your-domain.com/specialization_track/deployment/comments_tracking/`

### File Structure After Deployment

```
/var/www/html/specialization_track/
├── index.html                          # Main portal
├── review_dashboard.html               # Management dashboard
├── azure_solution_guide.html           # Complete Azure guide
├── deployment/
│   ├── comments_tracking/
│   │   ├── aws_comments.csv
│   │   ├── azure_comments.csv
│   │   └── review_summary.csv
│   ├── logs/
│   ├── aws_reviews/
│   ├── azure_reviews/
│   ├── deploy.sh
│   ├── backup_comments.sh
│   ├── nginx_deploy.sh
│   └── quick_start.sh
└── [other HTML files...]
```

### Comment Tracking Features

The deployment includes:
- ✅ **CSV-based comment system** for tracking all reviews
- ✅ **Automatic backups** of comment files
- ✅ **Web dashboard** for managing reviews
- ✅ **Export functionality** for comment data
- ✅ **Real-time monitoring** of learning progress

### Monitoring Commands

```bash
# Check nginx status
sudo systemctl status nginx

# Monitor access logs
sudo tail -f /var/log/nginx/specialization_track_access.log

# Check comment files
ls -la /var/www/html/specialization_track/deployment/comments_tracking/

# Run site monitoring
sudo /var/www/html/specialization_track/deployment/nginx_monitor.sh
```

### Backup Setup

The deployment includes automatic backup scripts. To set up regular backups:

```bash
# Add to crontab for automatic backups every 6 hours
sudo crontab -e

# Add this line:
0 */6 * * * /var/www/html/specialization_track/deployment/backup_comments.sh
```

### Troubleshooting

1. **Permission issues**: Ensure www-data owns the files
2. **404 errors**: Check nginx configuration and file paths
3. **CSV not downloading**: Verify nginx CSV handling configuration
4. **Comments not saving**: Check write permissions on comments_tracking folder

### Ready to Deploy?

Run this single command to deploy everything:

```bash
sudo /Users/niranjan/Downloads/specialization_track/deployment/nginx_deploy.sh
```

This will handle the entire deployment process automatically!