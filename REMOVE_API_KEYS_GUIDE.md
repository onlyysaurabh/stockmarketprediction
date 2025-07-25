# 🔒 Complete Guide: Remove API Keys from Git & GitHub History

## ⚠️ IMPORTANT: Backup First!
```bash
# Create a backup of your repository
cp -r /home/skylap/stockmarketprediction /home/skylap/stockmarketprediction-backup
```

## Method 1: Using git filter-repo (Recommended)

### Install git-filter-repo
```bash
# Option A: Via pip (in virtual environment)
python -m venv temp_env
source temp_env/bin/activate
pip install git-filter-repo

# Option B: Via system package manager (Arch Linux)
sudo pacman -S git-filter-repo

# Option C: Download directly
wget https://raw.githubusercontent.com/newren/git-filter-repo/main/git-filter-repo
chmod +x git-filter-repo
sudo mv git-filter-repo /usr/local/bin/
```

### Remove API Keys from History
```bash
cd /home/skylap/stockmarketprediction

# Remove specific API key patterns from all commits
git filter-repo --replace-text <(echo 'cvml8uhr01ql90pule5g***REMOVED***')
git filter-repo --replace-text <(echo 'd010qvhr01qv3oh20410***REMOVED***')
git filter-repo --replace-text <(echo '7for9)js*2g@04iyh3&#lpa8#_kyrfj58px)07fg1o558z$a^-***REMOVED***')

# Or remove entire lines containing API keys
git filter-repo --replace-text <(cat << 'EOF'
FINNHUB_API_KEYS=cvml8uhr01ql90pule5gcvml8uhr01ql90pule60,d010qvhr01qv3oh20410d010qvhr01qv3oh2041g,d010scpr01qv3oh20cd0d010scpr01qv3oh20cdg,d010si1r01qv3oh20dggd010si1r01qv3oh20dh0,d010tphr01qv3oh20kugd010tphr01qv3oh20kv0,d010tu1r01qv3oh20lq0d010tu1r01qv3oh20lqg,d010ug9r01qv3oh20p60d010ug9r01qv3oh20p6g,d010ukhr01qv3oh20q0gd010ukhr01qv3oh20q10,d010ut1r01qv3oh20rpgd010ut1r01qv3oh20rq0,d010v21r01qv3oh20sp0d010v21r01qv3oh20spg,d010v61r01qv3oh20tggd010v61r01qv3oh20th0***REMOVED***
SECRET_KEY = '7for9)js*2g@04iyh3&#lpa8#_kyrfj58px)07fg1o558z$a^-'***REMOVED***
EOF
)
```

## Method 2: Using BFG Repo-Cleaner (Alternative)

### Install BFG
```bash
# Download BFG jar file
wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar

# Create replacement file
cat > replace.txt << 'EOF'
cvml8uhr01ql90pule5g***REMOVED***
d010qvhr01qv3oh20410***REMOVED***
7for9)js*2g@04iyh3&#lpa8#_kyrfj58px)07fg1o558z$a^-***REMOVED***
EOF

# Run BFG
java -jar bfg-1.14.0.jar --replace-text replace.txt .git
```

## Method 3: Manual History Rewrite (For Specific Commits)

### If API keys are only in recent commits
```bash
# Interactive rebase to edit recent commits
git rebase -i HEAD~10  # Adjust number based on how far back

# For each commit with API keys, mark as 'edit'
# Then when rebasing stops:
# 1. Edit the files to remove API keys
# 2. git add .
# 3. git commit --amend --no-edit
# 4. git rebase --continue
```

## Step 2: Force Push to GitHub

⚠️ **WARNING**: This will overwrite GitHub history!

```bash
# Add the remote if it's missing after filter-repo
git remote add origin https://github.com/onlyysaurabh/stockmarketprediction.git

# Force push the cleaned history
git push --force-with-lease origin main

# If that fails, use force push (more dangerous)
git push --force origin main
```

## Step 3: Clean Up All Branches

```bash
# List all branches
git branch -a

# Clean each branch that might contain API keys
git checkout branch-name
git filter-repo --replace-text replace.txt
git push --force origin branch-name
```

## Step 4: Notify GitHub (Important!)

1. **Go to GitHub.com** → Your Repository → Settings → Security & analysis
2. **Enable secret scanning** if available
3. **Contact GitHub Support** to request a cache purge:
   ```
   Subject: Request to purge sensitive data from GitHub cache
   
   Repository: onlyysaurabh/stockmarketprediction
   Reason: Accidentally committed API keys, removed from history
   Commits affected: [list commit hashes if known]
   ```

## Step 5: Revoke Compromised API Keys

### Finnhub API Keys
1. Log in to [Finnhub.io](https://finnhub.io/)
2. Go to Dashboard → API Keys
3. **Delete all exposed API keys**
4. **Generate new API keys**
5. Update your `.env` file with new keys

### Django Secret Key
1. **Generate a new secret key**:
   ```python
   from django.core.management.utils import get_random_secret_key
   print(get_random_secret_key())
   ```
2. Update your `.env` file

## Step 6: Verify Cleanup

```bash
# Search for any remaining traces
git log --all --grep="cvml8uhr01ql90pule5g" --oneline
git log --all -S "cvml8uhr01ql90pule5g" --oneline
git log --all -S "FINNHUB_API_KEYS" --oneline

# Check specific files across history
git log --follow -p -- .env | grep -i "finnhub\|secret"
```

## Step 7: Protect Future Commits

### Update .gitignore (already done)
```bash
# Verify .env is ignored
grep -n ".env" .gitignore
```

### Add pre-commit hooks
```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
EOF

# Initialize
pre-commit install
pre-commit run --all-files
```

## Step 8: Monitor for Exposure

### Check if keys are indexed by search engines
- Search Google for your API keys
- Check GitHub search for your repository + API key fragments
- Monitor your Finnhub dashboard for unauthorized usage

## Recovery Commands (If Something Goes Wrong)

```bash
# Restore from backup
rm -rf /home/skylap/stockmarketprediction
mv /home/skylap/stockmarketprediction-backup /home/skylap/stockmarketprediction
cd /home/skylap/stockmarketprediction

# Or reset to remote state
git fetch origin
git reset --hard origin/main
```

## Final Checklist

- [ ] Backup created
- [ ] API keys removed from history
- [ ] Force pushed to GitHub
- [ ] All branches cleaned
- [ ] GitHub support contacted
- [ ] API keys revoked and regenerated
- [ ] New keys added to .env
- [ ] Cleanup verified
- [ ] Pre-commit hooks installed
- [ ] Search engines checked

---

**Remember**: Once sensitive data is on the internet, assume it's compromised forever. Always revoke and regenerate exposed credentials!
