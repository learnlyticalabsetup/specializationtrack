# Deployment Troubleshooting Guide
## assets.learnlytica.us/qreview/speializationtrack

### 🚨 **Current Issues**

1. **URL Typo**: `speializationtrack` should be `specializationtrack`
2. **Index.html not loading**: Likely a path or permission issue
3. **Web server configuration**: May need different setup than nginx script

### 🔧 **Quick Fixes**

#### **Fix 1: Correct the Directory Name**
```bash
# On your server, rename the directory
mv /qreview/speializationtrack /qreview/specializationtrack
```

#### **Fix 2: Check File Permissions**
```bash
# Set proper permissions for web access
chmod -R 755 /qreview/specializationtrack
chmod -R 777 /qreview/specializationtrack/deployment/comments_tracking

# If using Apache, ensure the web server can read files
chown -R apache:apache /qreview/specializationtrack
# OR if using nginx
chown -R nginx:nginx /qreview/specializationtrack
```

#### **Fix 3: Create a Simple .htaccess (if Apache)**
```bash
# Create .htaccess in your directory
cat > /qreview/specializationtrack/.htaccess << 'EOF'
DirectoryIndex index.html
Options +Indexes
<Files "*.csv">
    Header set Content-Type "text/csv"
    Header set Content-Disposition "attachment"
</Files>
EOF
```

### 🌐 **Correct URLs After Fix**

- **Main Portal**: `https://assets.learnlytica.us/qreview/specializationtrack/`
- **Review Dashboard**: `https://assets.learnlytica.us/qreview/specializationtrack/deployment/review_dashboard.html`
- **Azure Guide**: `https://assets.learnlytica.us/qreview/specializationtrack/azure_solution_guide.html`

### 🔍 **Debug Steps**

1. **Check if index.html exists**:
   ```bash
   ls -la /qreview/specializationtrack/index.html
   ```

2. **Test file access**:
   ```bash
   curl -I https://assets.learnlytica.us/qreview/specializationtrack/index.html
   ```

3. **Check web server logs**:
   ```bash
   tail -f /var/log/nginx/error.log
   # OR
   tail -f /var/log/apache2/error.log
   ```

### 📁 **Expected File Structure**
```
/qreview/specializationtrack/
├── index.html                    ← Main landing page
├── review_dashboard.html         
├── azure_solution_guide.html     
├── deployment/
│   ├── comments_tracking/
│   │   ├── aws_comments.csv
│   │   ├── azure_comments.csv
│   │   └── review_summary.csv
│   └── review_dashboard.html
└── js/
    └── comment-manager.js
```

### ⚡ **Quick Test**

Try accessing individual files:
1. `https://assets.learnlytica.us/qreview/specializationtrack/index.html`
2. `https://assets.learnlytica.us/qreview/specializationtrack/azure_solution_guide.html`

If these don't work, the issue is with file permissions or web server configuration.

### 🛠️ **Manual Deployment Commands**

Since you already copied the files, run these commands on your server:

```bash
# Navigate to your directory
cd /qreview

# Fix the directory name (if needed)
if [ -d "speializationtrack" ]; then
    mv speializationtrack specializationtrack
fi

# Set permissions
chmod -R 755 specializationtrack
chmod -R 777 specializationtrack/deployment/comments_tracking

# Test if index.html is readable
cat specializationtrack/index.html | head -5
```

### 📋 **Next Steps**

1. **Fix directory name** (speializationtrack → specializationtrack)
2. **Set proper file permissions**
3. **Test the corrected URL**: `https://assets.learnlytica.us/qreview/specializationtrack/`
4. **Check web server error logs** if still not working

The most likely issue is the typo in the directory name. Once fixed, your site should load properly!