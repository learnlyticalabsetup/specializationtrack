# Cloud Specialization Track - Deployment Guide

## 📋 Overview
This deployment package contains the complete Cloud Specialization Track for AWS and Azure, including:
- Practical assignments and solution guides
- MCQ questions and review systems
- Capstone projects
- Comment tracking and feedback systems

## 🚀 Deployment Instructions

### Prerequisites
- Web server (Apache, Nginx, or IIS)
- PHP support (optional, for dynamic features)
- SSL certificate (recommended for production)

### Quick Deployment

1. **Upload Files to Server**
   ```bash
   # Copy all files to your web server
   scp -r /Users/niranjan/Downloads/specialization_track/ user@your-server:/var/www/html/
   ```

2. **Set Permissions**
   ```bash
   chmod 755 /var/www/html/specialization_track/
   chmod 644 /var/www/html/specialization_track/*.html
   chmod 777 /var/www/html/specialization_track/deployment/comments_tracking/
   ```

3. **Configure Web Server**
   - Point domain to the `specialization_track` folder
   - Ensure `index.html` is set as default document
   - Enable directory browsing for CSV downloads

### 📊 Comment Tracking System

The system automatically tracks all review comments in CSV format:

#### AWS Comments (`aws_comments.csv`)
- Assignment reviews and feedback
- MCQ question improvements
- Solution guide enhancements
- Action items and resolutions

#### Azure Comments (`azure_comments.csv`)
- Practical lab feedback
- Code formatting reviews
- Step-by-step improvements
- Beginner-friendly suggestions

#### Summary Report (`review_summary.csv`)
- Overall platform statistics
- Completion percentages
- Average ratings
- Action item tracking

### 🔧 Server Configuration

#### Apache Configuration
```apache
<VirtualHost *:80>
    ServerName your-domain.com
    DocumentRoot /var/www/html/specialization_track
    DirectoryIndex index.html review_dashboard.html
    
    <Directory /var/www/html/specialization_track>
        Options Indexes FollowSymLinks
        AllowOverride All
        Require all granted
    </Directory>
    
    # Enable CSV downloads
    <Files "*.csv">
        Header set Content-Type "text/csv"
        Header set Content-Disposition "attachment"
    </Files>
</VirtualHost>
```

#### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;
    root /var/www/html/specialization_track;
    index index.html review_dashboard.html;
    
    location / {
        try_files $uri $uri/ =404;
    }
    
    location ~* \.csv$ {
        add_header Content-Type "text/csv";
        add_header Content-Disposition "attachment";
    }
}
```

### 📁 Directory Structure
```
specialization_track/
├── index.html (Main landing page)
├── review_dashboard.html (Review tracking dashboard)
├── aws_*.html (AWS content files)
├── azure_*.html (Azure content files)
├── *.csv (Question banks)
├── deployment/
│   ├── comments_tracking/
│   │   ├── aws_comments.csv
│   │   ├── azure_comments.csv
│   │   └── review_summary.csv
│   └── logs/
└── README.md
```

### 🔐 Security Considerations

1. **SSL/TLS**: Enable HTTPS for secure access
2. **Access Control**: Implement authentication for admin features
3. **File Permissions**: Restrict write access to comment files
4. **Backup**: Regular backups of CSV comment files

### 📈 Monitoring & Analytics

The deployment includes:
- Real-time comment tracking
- Review progress monitoring
- Automatic CSV generation
- Export functionality for all feedback

### 🚀 Production Deployment Checklist

- [ ] Server environment configured
- [ ] SSL certificate installed
- [ ] Domain DNS configured
- [ ] File permissions set correctly
- [ ] Comment tracking folders writable
- [ ] Backup system configured
- [ ] Monitoring tools installed
- [ ] Admin access credentials set

### 📞 Support & Maintenance

For deployment support:
1. Check server error logs
2. Verify file permissions
3. Test CSV download functionality
4. Monitor comment tracking system

### 🔄 Updates & Maintenance

To update the system:
1. Backup existing CSV files
2. Upload new content files
3. Preserve comment tracking data
4. Test all functionality

## 🌐 Access URLs

After deployment:
- Main Site: `https://your-domain.com/`
- Review Dashboard: `https://your-domain.com/review_dashboard.html`
- AWS Content: `https://your-domain.com/aws_index.html`
- Azure Content: `https://your-domain.com/azure_index.html`
- Comments Export: `https://your-domain.com/deployment/comments_tracking/`