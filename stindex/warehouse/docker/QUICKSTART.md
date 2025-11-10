# Quick Start Guide - STIndex Data Warehouse with Docker

## 1. Start the Database (One Command)

```bash
docker compose up -d
```

Wait ~10 seconds for the database to initialize.

## 2. Create Schema & Populate Data

```bash
# Create all tables
docker exec -i stindex-warehouse psql -U stindex -d stindex_warehouse < stindex/warehouse/schema/create_schema_docker.sql

# Populate reference data (dates, countries, taxonomies)
docker exec -i stindex-warehouse psql -U stindex -d stindex_warehouse < stindex/warehouse/schema/populate_dimensions.sql
```

## 3. Verify Installation

```bash
docker exec -it stindex-warehouse psql -U stindex -d stindex_warehouse

# In psql:
\dt  # List tables (should see 20+ tables)
SELECT COUNT(*) FROM dim_date;  # Should return ~18,628 dates (2000-2050)
\q   # Exit
```

## 4. Test with Python

```python
from stindex.warehouse.etl import DimensionalWarehouseETL

# Connect to warehouse
etl = DimensionalWarehouseETL(
    db_connection_string="postgresql://stindex:stindex@localhost:5432/stindex_warehouse"
)

# Test connection
etl.connect()
print("✓ Connected to warehouse!")
etl.disconnect()
```

## Connection Details

- **Host**: localhost
- **Port**: 5432
- **Database**: stindex_warehouse
- **Username**: stindex
- **Password**: stindex
- **Connection String**: `postgresql://stindex:stindex@localhost:5432/stindex_warehouse`

## pgAdmin (Optional GUI)

- **URL**: http://localhost:5050
- **Email**: admin@stindex.local
- **Password**: admin

## Common Commands

```bash
# View logs
docker compose logs -f stindex-warehouse

# Stop database
docker compose down

# Stop and delete all data (reset)
docker compose down -v

# Restart database
docker compose restart stindex-warehouse
```

## Next Steps

See:
- [docker/README.md](README.md) - Full Docker documentation
- [stindex/warehouse/README.md](../README.md) - Usage guide
- [stindex/warehouse/schema/README.md](../schema/README.md) - Schema details

## Troubleshooting

**Port 5432 already in use?**
```bash
# Edit docker-compose.yml, change port:
# ports:
#   - "5433:5432"

# Then connect with:
# postgresql://stindex:stindex@localhost:5433/stindex_warehouse
```

**Schema creation fails?**
```bash
# Reset schema
docker exec -it stindex-warehouse psql -U stindex -d stindex_warehouse \
  -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"

# Recreate
docker exec -i stindex-warehouse psql -U stindex -d stindex_warehouse \
  < stindex/warehouse/schema/create_schema_docker.sql
```
