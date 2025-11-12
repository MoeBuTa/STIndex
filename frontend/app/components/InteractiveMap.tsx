'use client'

import { useState, useMemo, useCallback } from 'react'
import Map, { Marker, Popup, Source, Layer } from 'react-map-gl'
import { Box, Badge, Text, VStack, HStack, Button, Slider, SliderTrack, SliderFilledTrack, SliderThumb } from '@chakra-ui/react'
import { EventCluster, SpatioTemporalEvent } from '../lib/analytics'
import 'mapbox-gl/dist/mapbox-gl.css'

interface BackendCluster {
  cluster_id: string
  cluster_type: string
  size: number
  centroid: {
    latitude: number
    longitude: number
    datetime: string
    time_range: {
      start: string
      end: string
    }
  }
  category_value: string | null
  dimension: string | null
  event_ids: string[]
}

interface BackendClusters {
  clusters: BackendCluster[]
  burst_periods: any[]
  statistics: any
}

interface InteractiveMapProps {
  events: SpatioTemporalEvent[]
  backendClusters?: BackendClusters | null
  height?: string
  showClusters?: boolean
  enableAnimation?: boolean
}

export function InteractiveMap({
  events,
  backendClusters,
  height = '600px',
  showClusters = true,
  enableAnimation = true,
}: InteractiveMapProps) {
  const [viewport, setViewport] = useState({
    longitude: 133.7751,
    latitude: -25.2744,
    zoom: 4,
  })
  const [selectedEvent, setSelectedEvent] = useState<SpatioTemporalEvent | null>(null)
  const [selectedCluster, setSelectedCluster] = useState<EventCluster | null>(null)
  const [timeFilter, setTimeFilter] = useState<number>(100) // Percentage 0-100

  // Get Mapbox token from environment or use placeholder
  const mapboxToken = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || 'YOUR_MAPBOX_TOKEN_HERE'

  // Filter events with valid coordinates
  const validEvents = useMemo(() => {
    return events.filter((e) => e.latitude && e.longitude)
  }, [events])

  // Transform backend clusters to the format expected by the component
  const clusters = useMemo<EventCluster[]>(() => {
    if (!showClusters || !backendClusters || !backendClusters.clusters) return []

    return backendClusters.clusters.map((bc) => {
      // Find events belonging to this cluster
      const clusterEvents = events.filter((e) => bc.event_ids.includes(e.id))

      return {
        id: bc.cluster_id,
        size: bc.size,
        centroid: {
          lat: bc.centroid.latitude,
          lng: bc.centroid.longitude,
        },
        timeRange: {
          start: bc.centroid.time_range.start,
          end: bc.centroid.time_range.end,
        },
        entities: new Set(bc.event_ids),
        dominantCategory: bc.category_value || undefined,
        events: clusterEvents,
        density: bc.size, // Use cluster size as a proxy for density
      }
    })
  }, [backendClusters, showClusters, events])

  // Filter events by time
  const filteredEvents = useMemo(() => {
    if (timeFilter === 100) return validEvents

    const sortedEvents = [...validEvents].sort((a, b) => {
      const dateA = a.timestamp || a.normalized_date || ''
      const dateB = b.timestamp || b.normalized_date || ''
      return dateA.localeCompare(dateB)
    })

    const cutoff = Math.floor((sortedEvents.length * timeFilter) / 100)
    return sortedEvents.slice(0, cutoff)
  }, [validEvents, timeFilter])

  // Fit map to events on mount
  const fitBounds = useCallback(() => {
    if (validEvents.length === 0) return

    const lats = validEvents.map((e) => e.latitude!)
    const lngs = validEvents.map((e) => e.longitude!)

    const minLat = Math.min(...lats)
    const maxLat = Math.max(...lats)
    const minLng = Math.min(...lngs)
    const maxLng = Math.max(...lngs)

    const centerLat = (minLat + maxLat) / 2
    const centerLng = (minLng + maxLng) / 2

    setViewport({
      longitude: centerLng,
      latitude: centerLat,
      zoom: 6,
    })
  }, [validEvents])

  // Heatmap data
  const heatmapGeoJSON = useMemo(() => {
    const features = filteredEvents.map((event) => ({
      type: 'Feature' as const,
      properties: {
        weight: 1,
      },
      geometry: {
        type: 'Point' as const,
        coordinates: [event.longitude!, event.latitude!],
      },
    }))

    return {
      type: 'FeatureCollection' as const,
      features,
    }
  }, [filteredEvents])

  if (mapboxToken === 'YOUR_MAPBOX_TOKEN_HERE') {
    return (
      <Box p={6} bg="gray.50" borderRadius="md" height={height}>
        <VStack spacing={4}>
          <Text fontSize="lg" fontWeight="bold">
            Map Configuration Required
          </Text>
          <Text>
            To enable the interactive map, please add your Mapbox access token:
          </Text>
          <Box p={4} bg="gray.100" borderRadius="md" fontFamily="mono" fontSize="sm">
            NEXT_PUBLIC_MAPBOX_TOKEN=your_token_here
          </Box>
          <Text fontSize="sm" color="gray.600">
            Get a free token at{' '}
            <Text as="span" color="blue.500" textDecoration="underline">
              https://account.mapbox.com/access-tokens/
            </Text>
          </Text>
        </VStack>
      </Box>
    )
  }

  return (
    <Box height={height} position="relative">
      <Map
        {...viewport}
        onMove={(evt) => setViewport(evt.viewState)}
        mapboxAccessToken={mapboxToken}
        style={{ width: '100%', height: '100%' }}
        mapStyle="mapbox://styles/mapbox/light-v11"
      >
        {/* Heatmap Layer */}
        {heatmapGeoJSON && (
          <Source id="heatmap" type="geojson" data={heatmapGeoJSON}>
            <Layer
              id="heatmap-layer"
              type="heatmap"
              paint={{
                'heatmap-weight': ['get', 'weight'],
                'heatmap-intensity': 1,
                'heatmap-color': [
                  'interpolate',
                  ['linear'],
                  ['heatmap-density'],
                  0,
                  'rgba(33,102,172,0)',
                  0.2,
                  'rgb(103,169,207)',
                  0.4,
                  'rgb(209,229,240)',
                  0.6,
                  'rgb(253,219,199)',
                  0.8,
                  'rgb(239,138,98)',
                  1,
                  'rgb(178,24,43)',
                ],
                'heatmap-radius': 20,
                'heatmap-opacity': 0.6,
              }}
            />
          </Source>
        )}

        {/* Story Arc Lines - Disabled: backend story arcs use different format */}
        {/* TODO: Transform backend story arcs to GeoJSON format for visualization */}

        {/* Event Markers */}
        {filteredEvents.map((event, idx) => (
          <Marker
            key={`${event.id}-${idx}`}
            longitude={event.longitude!}
            latitude={event.latitude!}
            anchor="bottom"
            onClick={(e) => {
              e.originalEvent.stopPropagation()
              setSelectedEvent(event)
              setSelectedCluster(null)
            }}
          >
            <Box
              w="12px"
              h="12px"
              borderRadius="full"
              bg={event.category === 'outbreak_report' ? 'red.500' : 'blue.500'}
              border="2px solid white"
              cursor="pointer"
              _hover={{ transform: 'scale(1.5)' }}
              transition="transform 0.2s"
            />
          </Marker>
        ))}

        {/* Cluster Markers */}
        {showClusters &&
          clusters.map((cluster) => (
            <Marker
              key={cluster.id}
              longitude={cluster.centroid.lng}
              latitude={cluster.centroid.lat}
              anchor="center"
              onClick={(e) => {
                e.originalEvent.stopPropagation()
                setSelectedCluster(cluster)
                setSelectedEvent(null)
              }}
            >
              <Box
                position="relative"
                w={`${Math.min(cluster.size * 8, 60)}px`}
                h={`${Math.min(cluster.size * 8, 60)}px`}
                borderRadius="full"
                bg="purple.500"
                opacity={0.6}
                border="2px solid white"
                cursor="pointer"
                display="flex"
                alignItems="center"
                justifyContent="center"
                _hover={{ opacity: 0.9 }}
              >
                <Text color="white" fontWeight="bold" fontSize="xs">
                  {cluster.size}
                </Text>
              </Box>
            </Marker>
          ))}

        {/* Event Popup */}
        {selectedEvent && (
          <Popup
            longitude={selectedEvent.longitude!}
            latitude={selectedEvent.latitude!}
            anchor="top"
            onClose={() => setSelectedEvent(null)}
            closeOnClick={false}
          >
            <VStack align="start" spacing={2} p={2}>
              <Text fontWeight="bold" fontSize="sm">
                {selectedEvent.text}
              </Text>
              {selectedEvent.category && (
                <Badge colorScheme="blue" fontSize="xs">
                  {selectedEvent.category}
                </Badge>
              )}
              {selectedEvent.timestamp && (
                <Text fontSize="xs" color="gray.600">
                  {selectedEvent.timestamp}
                </Text>
              )}
              <Text fontSize="xs" color="gray.500">
                {selectedEvent.source}
              </Text>
            </VStack>
          </Popup>
        )}

        {/* Cluster Popup */}
        {selectedCluster && (
          <Popup
            longitude={selectedCluster.centroid.lng}
            latitude={selectedCluster.centroid.lat}
            anchor="top"
            onClose={() => setSelectedCluster(null)}
            closeOnClick={false}
          >
            <VStack align="start" spacing={2} p={2} maxW="250px">
              <Text fontWeight="bold" fontSize="sm">
                Event Cluster
              </Text>
              <HStack>
                <Badge colorScheme="purple">{selectedCluster.size} events</Badge>
                {selectedCluster.dominantCategory && (
                  <Badge colorScheme="blue">{selectedCluster.dominantCategory}</Badge>
                )}
              </HStack>
              <Text fontSize="xs" color="gray.600">
                {selectedCluster.timeRange.start} to {selectedCluster.timeRange.end}
              </Text>
              <Text fontSize="xs" color="gray.500">
                Entities: {Array.from(selectedCluster.entities).slice(0, 3).join(', ')}
                {selectedCluster.entities.size > 3 && '...'}
              </Text>
            </VStack>
          </Popup>
        )}
      </Map>

      {/* Time Slider */}
      {enableAnimation && validEvents.length > 0 && (
        <Box
          position="absolute"
          bottom={4}
          left={4}
          right={4}
          bg="white"
          p={4}
          borderRadius="md"
          shadow="md"
        >
          <VStack spacing={2} align="stretch">
            <HStack justify="space-between">
              <Text fontSize="sm" fontWeight="medium">
                Timeline Filter
              </Text>
              <Text fontSize="xs" color="gray.600">
                Showing {filteredEvents.length} / {validEvents.length} events
              </Text>
            </HStack>
            <Slider
              value={timeFilter}
              onChange={setTimeFilter}
              min={0}
              max={100}
              step={1}
            >
              <SliderTrack>
                <SliderFilledTrack bg="blue.500" />
              </SliderTrack>
              <SliderThumb boxSize={6} />
            </Slider>
            <HStack justify="space-between">
              <Button
                size="xs"
                onClick={() => setTimeFilter(0)}
                variant="outline"
              >
                Reset
              </Button>
              <Button
                size="xs"
                onClick={() => setTimeFilter(100)}
                variant="outline"
              >
                Show All
              </Button>
              <Button size="xs" onClick={fitBounds} colorScheme="blue">
                Fit to Data
              </Button>
            </HStack>
          </VStack>
        </Box>
      )}

      {/* Legend */}
      <Box
        position="absolute"
        top={4}
        right={4}
        bg="white"
        p={3}
        borderRadius="md"
        shadow="md"
      >
        <VStack align="start" spacing={2}>
          <Text fontSize="sm" fontWeight="bold">
            Legend
          </Text>
          <HStack spacing={2}>
            <Box w="12px" h="12px" borderRadius="full" bg="blue.500" />
            <Text fontSize="xs">Event</Text>
          </HStack>
          <HStack spacing={2}>
            <Box w="12px" h="12px" borderRadius="full" bg="red.500" />
            <Text fontSize="xs">Outbreak</Text>
          </HStack>
          {showClusters && (
            <HStack spacing={2}>
              <Box
                w="20px"
                h="20px"
                borderRadius="full"
                bg="purple.500"
                opacity={0.6}
              />
              <Text fontSize="xs">Cluster</Text>
            </HStack>
          )}
        </VStack>
      </Box>

      {/* Stats Badge */}
      <Box position="absolute" top={4} left={4} bg="white" p={3} borderRadius="md" shadow="md">
        <VStack align="start" spacing={1}>
          <Text fontSize="xs" fontWeight="bold">
            Statistics
          </Text>
          <Text fontSize="xs">Events: {filteredEvents.length}</Text>
          {showClusters && <Text fontSize="xs">Clusters: {clusters.length}</Text>}
        </VStack>
      </Box>
    </Box>
  )
}
