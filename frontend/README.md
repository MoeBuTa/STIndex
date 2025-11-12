# STIndex - Spatiotemporal Information Extraction & Analysis Dashboard

An interactive web application for multi-dimensional spatiotemporal information extraction, visualization, and analysis. STIndex processes extracted data from documents containing temporal events, spatial locations, and custom dimensional information to provide rich visualizations and analytics for understanding patterns, bursts, and narrative progressions across time and space.

## Features

### 1. Interactive Dashboard

The main dashboard provides a comprehensive overview of extracted spatiotemporal data with multiple visualization modes:

- **Tabbed Interface**: Switch between 5 different visualization modes
- **Real-time Data Processing**: Client-side data transformation with optimized performance
- **Responsive Design**: Built with Chakra UI for seamless cross-device experience
- **Dynamic Component Loading**: Lazy-loaded visualizations for optimal performance

### 2. Overview Statistics

Get instant insights into your dataset with key metrics:

- **Document Statistics**: Total documents and chunks processed
- **Temporal Entities**: Count of dates, times, and durations extracted
- **Spatial Entities**: Count of locations and regions identified
- **Custom Dimensions**: Counts for entity types (event types, diseases, venues, etc.)

### 3. Advanced Analytics Panel

Comprehensive analytics for understanding data quality and patterns:

#### Extraction Quality Metrics
- **Relevance Score**: How pertinent extracted entities are to context (0-100%)
- **Accuracy Score**: Correctness of entity extraction (0-100%)
- **Completeness Score**: Coverage of information extraction (0-100%)
- **Consistency Score**: Internal coherence of extracted data (0-100%)

#### Event Burst Detection
- Automatically identifies temporal spikes in event activity
- Displays burst intensity and event counts
- Shows dominant locations and categories within bursts
- Configurable sliding window algorithm

#### Temporal Analytics
- Time span analysis (earliest to latest events)
- Cluster distribution metrics
- Average events per cluster calculations

#### Spatial Analytics
- Unique location tracking
- Geocoding success rates
- Geographic spread analysis
- Location type distribution

#### Dimension Statistics
- Breakdown by entity categories
- Top data sources identification
- Custom dimension-specific insights

### 4. Visualization Modes

#### 4.1 Basic Timeline
- **D3-based temporal visualization** with multi-track display
- Events grouped by category with color coding
- Interactive tooltips showing event details
- Temporal entities sorted by normalized dates
- Zoom and pan capabilities

#### 4.2 Dimension Breakdown
- **Custom dimension-specific analysis**
- Category frequency distribution charts
- Tabbed interface for different dimensions (event types, diseases, venues)
- Entity details with confidence scores and sources
- Filterable and sortable data tables

#### 4.3 Interactive Map
- **Mapbox-based geographic visualization**
- Heatmap layer showing event density
- Spatial clustering with backend-provided cluster data
- Time filter slider (0-100%) for temporal navigation
- Interactive markers for individual events and clusters
- Cluster details: centroid, size, dominant category, time range
- Interactive viewport controls (pan, zoom, rotate)
- Real-time cluster transformation and filtering

#### 4.4 Event Timeline
- **D3-powered multi-track timeline** showing:
  - Event points color-coded by category
  - Multiple category tracks for organized display
  - Horizontal and vertical scrolling for large datasets
- **Interactive event modals** with detailed information:
  - Quality scores (relevance, accuracy, completeness, consistency)
  - Source documents and URLs
  - Confidence metrics
  - Entity text and categories
- Hover tooltips with event previews
- Dynamic track sizing based on number of categories

#### 4.5 Entity Network
- **ReactFlow-based node-link diagram**
- Co-occurrence network graph showing entity relationships
- Nodes sized by entity frequency (top 20 entities)
- Dynamic color-coding by category (supports 10+ category types)
- Edges weighted by co-occurrence count within documents
- Interactive drag-and-drop visualization
- Minimap for navigation
- Legend showing top 10 entity types with counts
- Configurable minimum co-occurrence threshold

### 5. Advanced Algorithms

#### Spatiotemporal Clustering (DBSCAN-inspired)
- **Haversine distance** calculation for spatial proximity (km-based)
- **Temporal distance** calculation in days
- Configurable parameters:
  - Spatial radius (default: 50km)
  - Temporal window (default: 7 days)
  - Minimum points (default: 3)
- Computes cluster density and dominant entities
- Identifies spatiotemporal patterns

#### Burst Detection Algorithm
- Sliding window approach for identifying temporal spikes
- Customizable window parameters
- Burst intensity calculation
- Peak timing identification
- Dominant location and category analysis within bursts
- Visual representation with color-coded intensity

#### Co-Occurrence Network Analysis
- Groups entities by document context
- Tracks entity pairs appearing together
- Calculates co-occurrence frequencies
- Network visualization with weighted edges
- Top entity filtering by frequency

### 6. Data Quality & Reflection Scoring

Multi-dimensional quality assessment system:

- **Relevance**: Context pertinence of extracted entities
- **Accuracy**: Correctness of entity identification
- **Completeness**: Information coverage
- **Consistency**: Internal data coherence
- Optional reasoning fields for quality explanations
- Aggregated analytics across all events

### 7. Performance Optimizations

- **Dynamic imports** with SSR disabled for heavy visualization components
- **Memoization** of computed values using React hooks
- **Lazy loading** with loading spinners
- **Client-side hydration** prevention for SSR compatibility
- Efficient data transformation pipelines
- Optimized re-rendering strategies

### 8. Interactive Features

- **Time filtering** with slider controls for temporal navigation
- **Event modals** with comprehensive details on click
- **Tooltips** with hover information
- **Adjustable visualization parameters**:
  - Elevation scale for 3D visualizations
  - Radius scale for point sizes
  - Label visibility toggles
- **Tabbed navigation** between visualization modes
- **Responsive viewport controls** for maps
- **Drag-and-drop** for network graph nodes

## Technology Stack

### Core Framework
- **Next.js 16.0.1** - React framework for production
- **React 18.3.1** - UI library
- **TypeScript 5.9.3** - Type safety and developer experience

### UI & Styling
- **Chakra UI 2.10.9** - Responsive component library
- **Emotion** (react & styled) - CSS-in-JS styling
- **Framer Motion 10.18.0** - Animation library

### Visualization Libraries
- **D3.js 7.9.0** - Data-driven visualizations (timelines, charts)
- **Deck.gl 9.2.2** - Large-scale WebGL visualization
- **React Map GL 7.1.7** - Mapbox integration
- **Mapbox GL 3.8.0** - Interactive mapping
- **ReactFlow 11.11.4** - Node-link diagrams

### Utilities
- **React Icons 4.12.0** - Comprehensive icon library

## Data Structure

The application processes two main data sources:

### 1. Extraction Results (`extraction_results.json`)
Contains extracted entities with:
- **Document metadata**: ID, title, source URL
- **Temporal entities**: Dates, times, durations with normalization
- **Spatial entities**: Locations with coordinates and types
- **Custom dimensions**: Event types, diseases, venues, etc.
- **Reflection scores**: Quality metrics for each entity

### 2. Clusters (`clusters.json`)
Spatiotemporal clustering results with:
- Cluster ID, type, and size
- Centroid coordinates and datetime
- Time range (start/end)
- Category and dimension values
- Event IDs within cluster
- Burst periods with intensity metrics

## Getting Started

### Prerequisites
- Node.js 18+
- npm or yarn
- Mapbox API token (for map visualizations)

### Installation

```bash
# Install dependencies
npm install

# Set up environment variables
# Create .env.local file with:
NEXT_PUBLIC_MAPBOX_TOKEN=your_mapbox_token_here

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

### Data Setup

Place your data files in `/public/data/`:
- `extraction_results.json` - Main extracted entities (required)
- `clusters.json` - Spatiotemporal clustering results with burst periods (required)

## Use Cases

STIndex is particularly suited for:

- **Epidemiological Surveillance**: Disease outbreak tracking and pattern detection
- **Event Detection**: Identifying and analyzing temporal event patterns
- **Geographic Analysis**: Spatial pattern recognition and clustering
- **Multi-dimensional Data Exploration**: Custom dimension analysis (event types, categories, etc.)
- **Data Quality Assessment**: Extraction quality monitoring with reflection scores
- **Entity Relationship Analysis**: Understanding co-occurrence patterns between entities

## Project Structure

```
/app
  /components          # React components for visualizations
    - AnalyticsPanels.tsx
    - DashboardStats.tsx
    - DimensionBreakdown.tsx
    - EntityNetwork.tsx
    - InteractiveMap.tsx
    - StoryTimeline.tsx
    - TemporalTimeline.tsx
  /lib                 # Utility functions and algorithms
    - analytics.ts     # DBSCAN clustering, burst detection
  - page.tsx           # Main dashboard page
  - layout.tsx         # Root layout with metadata
  - providers.tsx      # Chakra UI provider
  - globals.css        # Global styles
/public/data           # Data files
```

## Configuration

### Next.js Configuration
- Standalone output mode for flexible deployment
- TypeScript strict mode enabled
- ES2017 target for broad compatibility

### Environment Variables
- `NEXT_PUBLIC_MAPBOX_TOKEN` - Mapbox API key for map visualizations

## Contributing

Contributions are welcome! This project is designed to be flexible and extensible for various spatiotemporal analysis needs.

## License

[Add your license information here]

## Acknowledgments

Built with modern web technologies and visualization libraries to provide powerful spatiotemporal data analysis capabilities.
