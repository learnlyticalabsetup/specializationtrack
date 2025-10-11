#!/usr/bin/env python3
"""
Update ML Specialization Index Page
Updates all assignment buttons from "Coming Soon" to active links
"""

import re

def update_index_page():
    """Update the index page to activate all assignment buttons"""
    filepath = "/Users/niranjan/Downloads/specialization_track/ml_specialization_assignments_index.html"
    
    print("Reading index file...")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace all "Coming Soon" buttons with active assignment links
    for i in range(6, 26):  # Assignments 6-25
        old_button = f'<button class="btn" disabled>🔒 Coming Soon</button>'
        new_button = f'<button class="btn" onclick="window.open(\'ml_practical_assignment_{i}.html\', \'_blank\')">📚 Start Assignment</button>'
        
        # Find the assignment number in context and replace
        # Look for the pattern with assignment number
        assignment_pattern = rf'(Assignment {i}:.*?){re.escape(old_button)}'
        replacement = rf'\1{new_button}'
        
        # Count replacements
        before_count = content.count(old_button)
        content = re.sub(assignment_pattern, replacement, content, flags=re.DOTALL)
        after_count = content.count(old_button)
        
        if before_count > after_count:
            print(f"✅ Updated Assignment {i} button")
    
    # Update the progress summary at the top
    # Find and update the progress statistics
    progress_pattern = r'(<div class="stat-number">)(\d+)(</div>\s*<div class="stat-label">Completed</div>)'
    content = re.sub(progress_pattern, r'\g<1>25\g<3>', content)
    
    # Update individual phase progress
    phases_complete = [
        (r'(\d+)/10 Complete', '10/10 Complete'),  # Phase 1
        (r'(\d+)/6 Complete', '6/6 Complete'),     # Phase 2  
        (r'(\d+)/4 Complete', '4/4 Complete'),     # Phase 3
        (r'(\d+)/5 Complete', '5/5 Complete'),     # Phase 4
    ]
    
    for pattern, replacement in phases_complete:
        content = re.sub(pattern, replacement, content)
    
    # Update progress bars to 100%
    progress_bars = [
        (r'(class="progress-fill phase1-progress" style="width: )\d+%', r'\g<1>100%'),
        (r'(class="progress-fill phase2-progress" style="width: )\d+%', r'\g<1>100%'),
        (r'(class="progress-fill phase3-progress" style="width: )\d+%', r'\g<1>100%'),
        (r'(class="progress-fill phase4-progress" style="width: )\d+%', r'\g<1>100%'),
    ]
    
    for pattern, replacement in progress_bars:
        content = re.sub(pattern, replacement, content)
    
    # Update the total completion message
    completion_pattern = r'(<h2>🎯 )(.*?)( Completion Status</h2>)'
    content = re.sub(completion_pattern, r'\g<1>100% All Assignments Complete!\g<3>', content)
    
    print("Writing updated content...")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Successfully updated index page!")
    print("🎉 All 25 assignments are now active and ready!")

if __name__ == "__main__":
    update_index_page()