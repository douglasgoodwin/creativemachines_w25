# Git Branches for Writing Projects: A Practical Guide

## Core Structure
```
main ------------------- (protected, final version)
  │
  └── rewrite --------- (your working draft)
      │
      └── analysis ---- (notes and thoughts)
```

## What Each Branch Does

### main
- Your "clean" branch
- Contains only complete, reviewed work
- Think of it as your portfolio version
```bash
# Only when your work is ready for others:
git checkout main
git merge rewrite
git push origin main
```

### rewrite
- Your "messy desk" where you do your work
- Create lots of commits here - they tell your story
- No need to keep it tidy
```bash
# Start your daily work:
git checkout rewrite
# Make changes to chapter.md
git add chapter.md
git commit -m "feat: updated photography examples with Instagram"
```

### analysis
- Your "notebook" with reading notes and ideas
- Preserves your thought process
- Good for looking back at your journey
```bash
# Capture your reading notes:
git checkout analysis
# Add to notes.md
git add notes.md
git commit -m "docs: thoughts on Benjamin's aura concept"
```

## Working With Others

When you need to work with other chapters, create a temporary collaboration branch:
```bash
# Create a branch to test ideas with Chapter 2
git checkout -b collab/chapter2
# Try out some changes
git add chapter.md
git commit -m "test: align reproduction concept with Ch 2"
```

If it works:
```bash
git checkout rewrite
git merge collab/chapter2
```

If it doesn't:
```bash
git checkout rewrite
git branch -D collab/chapter2  # Delete the test branch
```

## Common Problems and Solutions

### "Help! I made changes to the wrong branch!"
```bash
# Don't panic - Git can move your changes
git stash
git checkout correct-branch
git stash pop
```

### "I want to undo my last commit!"
```bash
# If you haven't pushed:
git reset --soft HEAD~1

# If you have pushed:
git revert HEAD
```

### "I need to see what changed!"
```bash
# See what you changed:
git diff

# See your recent work:
git log --oneline
```

### "My branch is a mess!"
```bash
# Start fresh from main:
git checkout main
git checkout -b rewrite-fresh
# Copy over what you want to keep
```

## Real Examples

### Example 1: Updating Examples
```bash
# Start your work
git checkout rewrite

# Edit chapter.md
# Change film examples to streaming examples

# Save your work
git add chapter.md
git commit -m "feat: replace cinema with Netflix examples

- Updated mass viewing section
- Added streaming statistics
- Connected to original theory"
```

### Example 2: Working with Chapter 2
```bash
# Create collaboration branch
git checkout -b collab/ch2-mass-media

# Edit chapter.md to align concepts
git add chapter.md
git commit -m "test: align mass media concepts with Ch 2"

# If it works:
git checkout rewrite
git merge collab/ch2-mass-media

# If you change your mind:
git checkout rewrite
git branch -D collab/ch2-mass-media
```

### Example 3: Preserving Analysis
```bash
git checkout analysis

# Add to reading-notes.md:
# "Benjamin's concept of aura seems especially 
#  relevant to NFTs because..."

git add reading-notes.md
git commit -m "docs: connections between aura and NFTs"
```

## Workshop Exercise Sequence

1. Basic Setup (15 min)
```bash
git init benjamin-chapter
cd benjamin-chapter
echo "# Chapter X: [Title]" > README.md
git add README.md
git commit -m "initial commit"
```

2. Create Working Structure (15 min)
```bash
# Create branches
git checkout -b rewrite
git checkout -b analysis

# Add initial files
touch chapter.md notes.md
```

3. Practice Daily Workflow (30 min)
- Make changes to chapter.md
- Commit changes
- Switch branches
- View history

4. Practice Collaboration (30 min)
- Create collaboration branch
- Make changes
- Merge or abandon changes
- Resolve simple conflicts

5. Recovery Practice (30 min)
- Fix wrong-branch commits
- Undo changes
- View history
- Restore old versions

## Tips for Success

1. Commit Often
- Each new example
- Each revised section
- Each new idea

2. Use Clear Messages
```bash
feat: update photography section with Instagram examples
docs: notes on how Instagram relates to Benjamin's aura
test: trying new connection with Chapter 4
```

3. Don't Fear Mistakes
- Git keeps everything
- You can always undo
- Ask for help early

4. Keep It Simple
- Stay on rewrite most of the time
- Only merge to main when ready
- Use collaboration branches sparingly
