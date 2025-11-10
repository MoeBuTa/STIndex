#!/bin/bash
# ==============================================================================
# PostgreSQL 15 Installation Script for User Directory
# ==============================================================================
# This script installs PostgreSQL 15 with pgvector and PostGIS extensions
# in the project's db/ directory without requiring root access.
#
# Prerequisites:
# - gcc, make, readline-devel, zlib-devel
# - Internet access for downloads
#
# Usage:
#   bash scripts/install_postgres.sh
# ==============================================================================

set -e

# Configuration
INSTALL_DIR="$(pwd)/db"
PG_VERSION="15.10"
PGVECTOR_VERSION="0.5.1"
PG_DATA="$INSTALL_DIR/data"
PG_LOG="$INSTALL_DIR/logs"
PG_PORT=5433  # Use non-standard port to avoid conflicts

echo "================================================================================"
echo "PostgreSQL $PG_VERSION Installation"
echo "================================================================================"
echo "Install directory: $INSTALL_DIR"
echo "Data directory: $PG_DATA"
echo "Port: $PG_PORT"
echo ""

# Create directories
mkdir -p "$INSTALL_DIR/src" "$PG_DATA" "$PG_LOG"

# ==============================================================================
# Step 1: Download and Install PostgreSQL
# ==============================================================================
echo "[1/5] Downloading PostgreSQL $PG_VERSION..."
cd "$INSTALL_DIR/src"

if [ ! -f "postgresql-$PG_VERSION.tar.gz" ]; then
    wget "https://ftp.postgresql.org/pub/source/v$PG_VERSION/postgresql-$PG_VERSION.tar.gz"
fi

echo "[2/5] Compiling PostgreSQL..."
if [ ! -d "postgresql-$PG_VERSION" ]; then
    tar -xzf "postgresql-$PG_VERSION.tar.gz"
fi

cd "postgresql-$PG_VERSION"

./configure \
    --prefix="$INSTALL_DIR" \
    --with-openssl \
    --with-readline \
    --with-zlib

make -j$(nproc)
make install

echo "✓ PostgreSQL installed to $INSTALL_DIR"

# ==============================================================================
# Step 2: Install pgvector Extension
# ==============================================================================
echo "[3/5] Installing pgvector extension..."
cd "$INSTALL_DIR/src"

if [ ! -d "pgvector" ]; then
    git clone --branch "v$PGVECTOR_VERSION" https://github.com/pgvector/pgvector.git
fi

cd pgvector

export PATH="$INSTALL_DIR/bin:$PATH"
export PG_CONFIG="$INSTALL_DIR/bin/pg_config"

make
make install

echo "✓ pgvector extension installed"

# ==============================================================================
# Step 3: Install PostGIS (Optional - requires GEOS, PROJ, GDAL)
# ==============================================================================
echo "[4/5] PostGIS installation..."
echo "Note: PostGIS installation is complex and requires GEOS, PROJ, GDAL libraries."
echo "Skipping PostGIS for now. You can install it later if needed."
echo "The warehouse will work without PostGIS, but spatial queries will be limited."

# ==============================================================================
# Step 4: Initialize Database
# ==============================================================================
echo "[5/5] Initializing database cluster..."

cd "$INSTALL_DIR"

if [ ! -f "$PG_DATA/PG_VERSION" ]; then
    ./bin/initdb -D "$PG_DATA" -U stindex --encoding=UTF8 --locale=en_US.UTF-8

    # Configure PostgreSQL
    cat >> "$PG_DATA/postgresql.conf" <<EOF

# Custom settings for STIndex warehouse
port = $PG_PORT
max_connections = 100
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 16MB
min_wal_size = 1GB
max_wal_size = 4GB

# Logging
logging_collector = on
log_directory = '$PG_LOG'
log_filename = 'postgresql-%Y-%m-%d.log'
log_statement = 'all'
EOF

    # Allow local connections
    cat >> "$PG_DATA/pg_hba.conf" <<EOF

# Local connections for stindex user
local   all             stindex                                 trust
host    all             stindex         127.0.0.1/32            trust
host    all             stindex         ::1/128                 trust
EOF

    echo "✓ Database cluster initialized"
else
    echo "✓ Database cluster already exists"
fi

# ==============================================================================
# Create Helper Scripts
# ==============================================================================
echo "Creating helper scripts..."

# Start script
cat > "$INSTALL_DIR/start_postgres.sh" <<'EOF'
#!/bin/bash
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$INSTALL_DIR/bin:$PATH"

echo "Starting PostgreSQL..."
pg_ctl -D "$INSTALL_DIR/data" -l "$INSTALL_DIR/logs/postgres.log" start

# Wait for server to start
sleep 2

# Check status
pg_ctl -D "$INSTALL_DIR/data" status

echo ""
echo "PostgreSQL is running!"
echo "Connection string: postgresql://stindex@localhost:5433/stindex_warehouse"
echo ""
echo "To stop: $INSTALL_DIR/stop_postgres.sh"
EOF
chmod +x "$INSTALL_DIR/start_postgres.sh"

# Stop script
cat > "$INSTALL_DIR/stop_postgres.sh" <<'EOF'
#!/bin/bash
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$INSTALL_DIR/bin:$PATH"

echo "Stopping PostgreSQL..."
pg_ctl -D "$INSTALL_DIR/data" stop
EOF
chmod +x "$INSTALL_DIR/stop_postgres.sh"

# Status script
cat > "$INSTALL_DIR/status_postgres.sh" <<'EOF'
#!/bin/bash
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$INSTALL_DIR/bin:$PATH"

pg_ctl -D "$INSTALL_DIR/data" status
EOF
chmod +x "$INSTALL_DIR/status_postgres.sh"

# Create database script
cat > "$INSTALL_DIR/create_warehouse.sh" <<'EOF'
#!/bin/bash
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$INSTALL_DIR")"
export PATH="$INSTALL_DIR/bin:$PATH"

echo "Creating stindex_warehouse database..."
createdb -U stindex -p 5433 stindex_warehouse

echo "Installing extensions..."
psql -U stindex -p 5433 -d stindex_warehouse <<SQL
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
-- PostGIS would be: CREATE EXTENSION IF NOT EXISTS postgis;
SQL

echo "Creating schema..."
psql -U stindex -p 5433 -d stindex_warehouse < "$PROJECT_ROOT/stindex/warehouse/schema/create_schema.sql"

echo "Populating dimensions..."
psql -U stindex -p 5433 -d stindex_warehouse < "$PROJECT_ROOT/stindex/warehouse/schema/populate_dimensions.sql"

echo "✓ Warehouse created successfully!"
echo "Connection: postgresql://stindex@localhost:5433/stindex_warehouse"
EOF
chmod +x "$INSTALL_DIR/create_warehouse.sh"

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "================================================================================"
echo "Installation Complete!"
echo "================================================================================"
echo ""
echo "PostgreSQL $PG_VERSION installed to: $INSTALL_DIR"
echo "Data directory: $PG_DATA"
echo "Port: $PG_PORT"
echo ""
echo "Next steps:"
echo "  1. Start PostgreSQL:"
echo "     $INSTALL_DIR/start_postgres.sh"
echo ""
echo "  2. Create warehouse:"
echo "     $INSTALL_DIR/create_warehouse.sh"
echo ""
echo "  3. Update cfg/warehouse.yml with:"
echo "     connection_string: \"postgresql://stindex@localhost:$PG_PORT/stindex_warehouse\""
echo ""
echo "  4. Check status anytime:"
echo "     $INSTALL_DIR/status_postgres.sh"
echo ""
echo "  5. Stop PostgreSQL:"
echo "     $INSTALL_DIR/stop_postgres.sh"
echo ""
echo "================================================================================"
