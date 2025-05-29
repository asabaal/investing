# Database Configuration

## Storage Location

The market data database is now stored **outside the repository** to avoid git issues with large files.

### Default Location
```
~/.market_data/market_data.db
```

### Why This Location?
- **Git-friendly**: Database files are excluded from version control
- **User-specific**: Each user has their own database copy
- **Persistent**: Survives repository clones and updates
- **Standard**: Follows OS conventions for user data

## Database Path Resolution

The system automatically handles database location:

1. **Default Behavior**: Uses `~/.market_data/market_data.db`
2. **Custom Path**: Can specify custom path when needed
3. **Auto-creation**: Directory is created automatically if needed

## Code Usage

### Automatic (Recommended)
```python
# Uses ~/.market_data/market_data.db automatically
db = MarketDataDatabase()
```

### Custom Path
```python
# Use custom location if needed
db = MarketDataDatabase(db_path="/path/to/custom/database.db")
```

### Environment Variable (Optional)
```bash
# Optional: Set custom default via environment
export MARKET_DATA_DB_PATH="/custom/path/market_data.db"
```

## Migration Status

✅ **Database moved**: From `./market_data.db` to `~/.market_data/market_data.db`  
✅ **Code updated**: All components now use new default location  
✅ **Git cleaned**: Database files excluded from repository  
✅ **Backward compatible**: Custom paths still supported  

## File Structure

```
~/.market_data/
├── market_data.db          # Main database (190MB)
├── backups/                # Optional backup location
└── logs/                   # Optional log storage
```

## Benefits

- **Repository size**: Dramatically reduced (no 190MB database)
- **Git performance**: Faster clones, pushes, pulls
- **Data persistence**: Database survives repo operations
- **Multi-user**: Each user maintains separate data
- **Backup friendly**: Database in predictable location

## Dashboard Integration

The admin dashboard automatically works with the new location:

```bash
# Launch dashboard (uses new database location)
python launch_dashboard.py

# Visit: http://localhost:5000
```

## Troubleshooting

### Database Not Found
If you get database errors:

1. **Check location**: `ls -la ~/.market_data/`
2. **Verify permissions**: `ls -la ~/.market_data/market_data.db`
3. **Regenerate if needed**: The system can create a new database

### Migration Issues
If you need to move existing database:

```bash
# Copy from repo to new location
cp ./market_data.db ~/.market_data/market_data.db

# Or move if you want to clean repo immediately
mv ./market_data.db ~/.market_data/market_data.db
```

### Custom Locations
For team environments or custom setups:

```python
# Point to shared network location
db = MarketDataDatabase(db_path="/shared/market_data/database.db")

# Or use relative path for testing
db = MarketDataDatabase(db_path="./test_data.db")
```

---

**Note**: This change makes the repository much more git-friendly while keeping all functionality intact!