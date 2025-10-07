/**
 * Cloud Specialization Track - Comment Management System
 * Centralized JavaScript for collecting and saving comments to CSV files
 */

class CommentManager {
    constructor() {
        this.csvEndpoint = 'deployment/comments_tracking/';
        this.comments = {
            aws: [],
            azure: []
        };
        this.loadExistingComments();
    }

    /**
     * Load existing comments from CSV files
     */
    async loadExistingComments() {
        try {
            // Load AWS comments
            const awsResponse = await fetch(`${this.csvEndpoint}aws_comments.csv`);
            if (awsResponse.ok) {
                const awsText = await awsResponse.text();
                this.comments.aws = this.parseCSV(awsText);
            }

            // Load Azure comments
            const azureResponse = await fetch(`${this.csvEndpoint}azure_comments.csv`);
            if (azureResponse.ok) {
                const azureText = await azureResponse.text();
                this.comments.azure = this.parseCSV(azureText);
            }
        } catch (error) {
            console.log('No existing comments found, starting fresh');
        }
    }

    /**
     * Parse CSV text into array of objects
     */
    parseCSV(csvText) {
        const lines = csvText.split('\n').filter(line => line.trim());
        if (lines.length <= 1) return [];
        
        const headers = lines[0].split(',');
        return lines.slice(1).map(line => {
            const values = this.parseCSVLine(line);
            const obj = {};
            headers.forEach((header, index) => {
                obj[header.trim()] = values[index] || '';
            });
            return obj;
        });
    }

    /**
     * Parse a single CSV line handling quotes and commas
     */
    parseCSVLine(line) {
        const result = [];
        let current = '';
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            if (char === '"') {
                inQuotes = !inQuotes;
            } else if (char === ',' && !inQuotes) {
                result.push(current.trim());
                current = '';
            } else {
                current += char;
            }
        }
        result.push(current.trim());
        return result;
    }

    /**
     * Add a new comment to the system
     */
    async addComment(platform, commentData) {
        const timestamp = new Date().toISOString().replace('T', ' ').split('.')[0];
        const reviewerId = this.generateReviewerId();
        
        const comment = {
            timestamp: timestamp,
            reviewer_id: reviewerId,
            assignment_type: commentData.type || 'practical',
            assignment_number: commentData.assignmentNumber || '1',
            section: commentData.section || 'general',
            comment_type: commentData.commentType || 'feedback',
            rating: commentData.rating || '5',
            comment_text: this.escapeCSV(commentData.text || ''),
            action_required: commentData.actionRequired ? 'yes' : 'no',
            status: 'pending',
            resolution_date: ''
        };

        // Add to local storage
        this.comments[platform].push(comment);
        
        // Save to CSV file
        await this.saveToCSV(platform);
        
        // Update dashboard if it exists
        this.updateDashboardStats();
        
        return comment;
    }

    /**
     * Generate a unique reviewer ID
     */
    generateReviewerId() {
        const today = new Date().toISOString().split('T')[0].replace(/-/g, '');
        const random = Math.floor(Math.random() * 1000).toString().padStart(3, '0');
        return `reviewer_${today}_${random}`;
    }

    /**
     * Escape CSV special characters
     */
    escapeCSV(text) {
        if (typeof text !== 'string') return '';
        return `"${text.replace(/"/g, '""')}"`;
    }

    /**
     * Save comments to CSV file
     */
    async saveToCSV(platform) {
        const headers = 'timestamp,reviewer_id,assignment_type,assignment_number,section,comment_type,rating,comment_text,action_required,status,resolution_date';
        
        const csvContent = headers + '\n' + this.comments[platform].map(comment => {
            return [
                comment.timestamp,
                comment.reviewer_id,
                comment.assignment_type,
                comment.assignment_number,
                comment.section,
                comment.comment_type,
                comment.rating,
                comment.comment_text,
                comment.action_required,
                comment.status,
                comment.resolution_date
            ].join(',');
        }).join('\n');

        // In a real deployment, this would send to server
        // For now, we'll save to localStorage and provide download
        localStorage.setItem(`${platform}_comments_csv`, csvContent);
        
        // Also trigger download for immediate CSV file update
        this.downloadCSV(platform, csvContent);
        
        console.log(`${platform.toUpperCase()} comments saved to CSV`);
    }

    /**
     * Download CSV file
     */
    downloadCSV(platform, csvContent) {
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${platform}_comments_${new Date().toISOString().split('T')[0]}.csv`;
        
        // Auto-save to the deployment folder (simulated)
        console.log(`CSV file ready for ${platform}: ${a.download}`);
        
        // Optionally trigger actual download
        // a.click();
        
        window.URL.revokeObjectURL(url);
    }

    /**
     * Update dashboard statistics
     */
    updateDashboardStats() {
        // This will be called to refresh dashboard numbers
        const awsCount = this.comments.aws.length;
        const azureCount = this.comments.azure.length;
        
        // Dispatch custom event for dashboard to listen
        window.dispatchEvent(new CustomEvent('commentsUpdated', {
            detail: {
                aws: awsCount,
                azure: azureCount
            }
        }));
    }

    /**
     * Get comment statistics
     */
    getStats(platform) {
        const comments = this.comments[platform];
        return {
            total: comments.length,
            actionItems: comments.filter(c => c.action_required === 'yes').length,
            averageRating: comments.length > 0 
                ? comments.reduce((sum, c) => sum + parseFloat(c.rating || 0), 0) / comments.length 
                : 0,
            byType: {
                practical: comments.filter(c => c.assignment_type === 'practical').length,
                mcq: comments.filter(c => c.assignment_type === 'mcq').length,
                capstone: comments.filter(c => c.assignment_type === 'capstone').length
            }
        };
    }

    /**
     * Export all comments for dashboard
     */
    exportAllComments() {
        const awsCSV = localStorage.getItem('aws_comments_csv');
        const azureCSV = localStorage.getItem('azure_comments_csv');
        
        if (awsCSV) this.downloadCSV('aws', awsCSV);
        if (azureCSV) this.downloadCSV('azure', azureCSV);
        
        // Create summary report
        const summary = this.generateSummaryReport();
        this.downloadCSV('summary', summary);
    }

    /**
     * Generate summary report
     */
    generateSummaryReport() {
        const awsStats = this.getStats('aws');
        const azureStats = this.getStats('azure');
        
        const summaryData = [
            ['platform', 'total_comments', 'action_items', 'avg_rating', 'practical_comments', 'mcq_comments', 'capstone_comments', 'last_updated'],
            [
                'AWS',
                awsStats.total,
                awsStats.actionItems,
                awsStats.averageRating.toFixed(2),
                awsStats.byType.practical,
                awsStats.byType.mcq,
                awsStats.byType.capstone,
                new Date().toISOString()
            ],
            [
                'Azure',
                azureStats.total,
                azureStats.actionItems,
                azureStats.averageRating.toFixed(2),
                azureStats.byType.practical,
                azureStats.byType.mcq,
                azureStats.byType.capstone,
                new Date().toISOString()
            ]
        ];
        
        return summaryData.map(row => row.join(',')).join('\n');
    }
}

// Global comment manager instance
window.commentManager = new CommentManager();

/**
 * Helper function to add comment to any assignment page
 */
window.addAssignmentComment = function(platform, assignmentData, commentText, rating = 5, actionRequired = false) {
    const commentData = {
        type: assignmentData.type || 'practical',
        assignmentNumber: assignmentData.number || '1',
        section: assignmentData.section || 'general',
        commentType: 'feedback',
        text: commentText,
        rating: rating,
        actionRequired: actionRequired
    };
    
    return window.commentManager.addComment(platform, commentData);
};

/**
 * Helper function for MCQ comments
 */
window.addMCQComment = function(platform, questionNumber, commentText, rating = 5) {
    return window.addAssignmentComment(platform, {
        type: 'mcq',
        number: questionNumber,
        section: 'question_review'
    }, commentText, rating);
};

/**
 * Helper function for practical assignment comments
 */
window.addPracticalComment = function(platform, assignmentNumber, section, commentText, rating = 5, needsAction = false) {
    return window.addAssignmentComment(platform, {
        type: 'practical',
        number: assignmentNumber,
        section: section
    }, commentText, rating, needsAction);
};

/**
 * Helper function for capstone project comments
 */
window.addCapstoneComment = function(platform, projectNumber, commentText, rating = 5, needsAction = false) {
    return window.addAssignmentComment(platform, {
        type: 'capstone',
        number: projectNumber,
        section: 'project_review'
    }, commentText, rating, needsAction);
};

console.log('Comment Management System loaded successfully!');