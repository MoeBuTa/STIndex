# STIndex Data Warehouse - Docker Setup

## Overview

This directory contains Docker configuration for running the STIndex data warehouse with PostgreSQL 15, pgvector, and PostGIS.

## Quick Start

### 1. Start the Database

```bash
# Start PostgreSQL + pgAdmin
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f stindex-warehouse
```

The database will be available at `localhost:5432` with:
- **Database**: `stindex_warehouse`
- **Username**: `stindex`
- **Password**: `stindex`

### 2. Create the Schema

```bash
# Wait for database to be healthy (check with docker compose ps)
# Then create the warehouse schema

docker exec -i stindex-warehouse psql -U stindex -d stindex_warehouse < stindex/warehouse/schema/create_schema_docker.sql
```

### 3. Populate Reference Data

```bash
# Pre-populate dates (2000-2050), countries, continents, taxonomies
docker exec -i stindex-warehouse psql -U stindex -d stindex_warehouse < stindex/warehouse/schema/populate_dimensions.sql
```

### 4. Verify Installation

```bash
# Connect to database
docker exec -it stindex-warehouse psql -U stindex -d stindex_warehouse

# In psql, run:
# Check extensions
SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'postgis', 'pg_trgm');

# Check tables
\dt

# Check table counts
SELECT 'dim_date' AS table_name, COUNT(*) FROM dim_date
UNION ALL SELECT 'dim_country', COUNT(*) FROM dim_country
UNION ALL SELECT 'dim_state', COUNT(*) FROM dim_state;

# Exit psql
\q
```

## Services

### PostgreSQL Warehouse (`stindex-warehouse`)

- **Image**: `pgvector/pgvector:pg15` (includes pgvector extension)
- **Port**: 5432
- **Database**: `stindex_warehouse`
- **Extensions**: vector, postgis, pg_trgm

**Connection String**:
```
postgresql://stindex:stindex@localhost:5432/stindex_warehouse
```

### pgAdmin (Optional GUI)

- **Port**: 5050
- **URL**: http://localhost:5050
- **Email**: `admin@stindex.local`
- **Password**: `admin`

To connect to the warehouse in pgAdmin:
1. Open http://localhost:5050
2. Login with credentials above
3. Add New Server:
   - **Name**: STIndex Warehouse
   - **Host**: stindex-warehouse (or host.docker.internal on Mac/Windows)
   - **Port**: 5432
   - **Database**: stindex_warehouse
   - **Username**: stindex
   - **Password**: stindex

## Configuration

### Update Connection String in Config

Edit `cfg/warehouse.yml`:

```yaml
database:
  connection_string: "postgresql://stindex:stindex@localhost:5432/stindex_warehouse"
```

Or for Python code:

```python
from stindex.warehouse.etl import DimensionalWarehouseETL

etl = DimensionalWarehouseETL(
    db_connection_string="postgresql://stindex:stindex@localhost:5432/stindex_warehouse"
)
```

## Docker Commands

### Start Services

```bash
# Start all services
docker compose up -d

# Start only database (no pgAdmin)
docker compose up -d stindex-warehouse
```

### Stop Services

```bash
# Stop all services
docker compose down

# Stop and remove volumes (WARNING: deletes all data)
docker compose down -v
```

### View Logs

```bash
# All services
docker compose logs -f

# Just database
docker compose logs -f stindex-warehouse

# Just pgAdmin
docker compose logs -f pgadmin
```

### Database Shell

```bash
# Connect to psql
docker exec -it stindex-warehouse psql -U stindex -d stindex_warehouse

# Run SQL file
docker exec -i stindex-warehouse psql -U stindex -d stindex_warehouse < file.sql

# Dump database
docker exec stindex-warehouse pg_dump -U stindex stindex_warehouse > backup.sql

# Restore database
docker exec -i stindex-warehouse psql -U stindex -d stindex_warehouse < backup.sql
```

### Container Management

```bash
# Restart database
docker compose restart stindex-warehouse

# View container stats
docker stats stindex-warehouse

# Inspect container
docker inspect stindex-warehouse
```

## Data Persistence

Data is persisted in Docker volumes:

- `stindex-warehouse-data`: PostgreSQL data directory
- `stindex-pgadmin-data`: pgAdmin configuration

### Backup and Restore

```bash
# Backup volume
docker run --rm -v stindex-warehouse-data:/data -v $(pwd):/backup \
  ubuntu tar czf /backup/warehouse-backup.tar.gz /data

# Restore volume
docker run --rm -v stindex-warehouse-data:/data -v $(pwd):/backup \
  ubuntu tar xzf /backup/warehouse-backup.tar.gz -C /
```

## Troubleshooting

### Database Won't Start

```bash
# Check logs
docker compose logs stindex-warehouse

# Check if port 5432 is already in use
lsof -i :5432

# If port is in use, edit docker-compose.yml to use different port:
# ports:
#   - "5433:5432"
```

### Extensions Not Installed

```bash
# Manually install extensions
docker exec -it stindex-warehouse psql -U stindex -d stindex_warehouse

# In psql:
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

### Schema Creation Fails

```bash
# Drop and recreate schema
docker exec -it stindex-warehouse psql -U stindex -d stindex_warehouse -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"

# Recreate schema
docker exec -i stindex-warehouse psql -U stindex -d stindex_warehouse < stindex/warehouse/schema/create_schema_docker.sql
```

### Reset Everything

```bash
# Stop and remove all containers and volumes
docker compose down -v

# Remove images (optional)
docker rmi pgvector/pgvector:pg15 dpage/pgadmin4

# Start fresh
docker compose up -d
```

## Performance Tuning

For production, edit `docker-compose.yml` to adjust PostgreSQL settings:

```yaml
environment:
  POSTGRES_SHARED_BUFFERS: 512MB      # 25% of RAM
  POSTGRES_WORK_MEM: 32MB             # For sorting/joins
  POSTGRES_MAINTENANCE_WORK_MEM: 128MB
  POSTGRES_EFFECTIVE_CACHE_SIZE: 2GB  # 50-75% of RAM
  POSTGRES_MAX_CONNECTIONS: 200
```

Or create a custom `postgresql.conf`:

```bash
# Uncomment in docker-compose.yml:
# - ./docker/postgresql.conf:/etc/postgresql/postgresql.conf
```

## Resource Limits

Limit CPU and memory usage (uncomment in docker-compose.yml):

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
    reservations:
      cpus: '1.0'
      memory: 2G
```

## Network Configuration

The services run on a dedicated network `stindex-network`. To connect other services:

```yaml
your-service:
  networks:
    - stindex-network

networks:
  stindex-network:
    external: true
```

## Security Notes

**⚠️ Default credentials are for development only!**

For production:

1. Change default passwords in `docker-compose.yml`
2. Use Docker secrets or environment variables
3. Enable SSL/TLS for PostgreSQL
4. Restrict network access
5. Use read-only volumes where possible

## Next Steps

After setup:

1. ✅ Database running with extensions
2. ✅ Schema created
3. ✅ Reference data populated
4. 🚀 Start using the warehouse:

```python
from stindex.warehouse.etl import DimensionalWarehouseETL

etl = DimensionalWarehouseETL(
    db_connection_string="postgresql://stindex:stindex@localhost:5432/stindex_warehouse"
)

# Load extraction results
etl.load_extraction_results(results, metadata, document_text)
```

See [stindex/warehouse/README.md](../README.md) for usage examples.
