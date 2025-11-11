'use client'

import { useMemo, useState, useEffect, useRef } from 'react'
import { Box, VStack, HStack, Text, Select, Switch, FormControl, FormLabel, Slider, SliderTrack, SliderFilledTrack, SliderThumb } from '@chakra-ui/react'
import DeckGL from '@deck.gl/react'
import { ColumnLayer } from '@deck.gl/layers'
import { COORDINATE_SYSTEM } from '@deck.gl/core'
import type { MapViewState } from '@deck.gl/core'
import { SpatioTemporalEvent } from '../lib/analytics'

interface SpaceTimeCubeProps {
  events: SpatioTemporalEvent[]
  height?: string
}

const CATEGORY_COLORS: { [key: string]: [number, number, number] } = {
  'risk_warning': [255, 99, 71],      // Red
  'outbreak_report': [255, 165, 0],   // Orange
  'surveillance': [65, 105, 225],     // Blue
  'measles': [255, 215, 0],           // Gold
  'tuberculosis': [147, 112, 219],    // Purple
  'polio': [50, 205, 50],             // Green
  'unknown': [169, 169, 169],         // Gray
}

export function SpaceTimeCube({ events, height = '700px' }: SpaceTimeCubeProps) {
  const [isMounted, setIsMounted] = useState(false)
  const [webGLSupported, setWebGLSupported] = useState(true)
  const [deckGLReady, setDeckGLReady] = useState(false)
  const [initError, setInitError] = useState<string | null>(null)
  const [hasSize, setHasSize] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const deckGLKeyRef = useRef(0)

  const [viewState, setViewState] = useState<MapViewState>({
    longitude: 0,
    latitude: 20,
    zoom: 1,
    pitch: 45,
    bearing: 0,
    minZoom: 0,
    maxZoom: 10,
    minPitch: 0,
    maxPitch: 85,
  })

  const [showLabels, setShowLabels] = useState(true)
  const [elevationScale, setElevationScale] = useState(50000)
  const [radiusScale, setRadiusScale] = useState(10000)

  // Ensure component only renders on client side and check WebGL support
  useEffect(() => {
    // Check if WebGL is supported
    try {
      const canvas = document.createElement('canvas')
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl')
      if (!gl) {
        setWebGLSupported(false)
        setIsMounted(true)
        return
      }
    } catch (e) {
      setWebGLSupported(false)
      setIsMounted(true)
      return
    }

    setIsMounted(true)
  }, [])

  // Wait for container to have proper dimensions before initializing DeckGL
  useEffect(() => {
    if (!isMounted || !containerRef.current) return

    const checkSize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect()
        if (rect.width > 0 && rect.height > 0) {
          setHasSize(true)
          // Additional delay after size is confirmed
          setTimeout(() => {
            setDeckGLReady(true)
          }, 200)
        }
      }
    }

    // Check size immediately
    checkSize()

    // Also check after a short delay in case of layout shifts
    const timer = setTimeout(checkSize, 100)

    return () => clearTimeout(timer)
  }, [isMounted])

  // Filter events with both spatial and temporal data
  const spatioTemporalEvents = useMemo(() => {
    return events.filter(
      (e) =>
        e.latitude !== undefined &&
        e.longitude !== undefined &&
        (e.timestamp || e.normalized_date)
    )
  }, [events])

  // Normalize temporal data to elevation
  const processedEvents = useMemo(() => {
    if (spatioTemporalEvents.length === 0) return []

    // Extract timestamps
    const timestamps = spatioTemporalEvents
      .map((e) => {
        const dateStr = e.timestamp || e.normalized_date || ''
        try {
          // Handle different date formats
          if (dateStr.match(/^\d{4}$/)) {
            return new Date(`${dateStr}-01-01`).getTime()
          } else if (dateStr.match(/^\d{4}-\d{2}$/)) {
            return new Date(`${dateStr}-01`).getTime()
          } else {
            return new Date(dateStr).getTime()
          }
        } catch {
          return null
        }
      })
      .filter((t): t is number => t !== null)

    const minTime = Math.min(...timestamps)
    const maxTime = Math.max(...timestamps)
    const timeRange = maxTime - minTime || 1

    // Process events with normalized elevation
    return spatioTemporalEvents.map((e, idx) => {
      const dateStr = e.timestamp || e.normalized_date || ''
      let timestamp = 0
      try {
        if (dateStr.match(/^\d{4}$/)) {
          timestamp = new Date(`${dateStr}-01-01`).getTime()
        } else if (dateStr.match(/^\d{4}-\d{2}$/)) {
          timestamp = new Date(`${dateStr}-01`).getTime()
        } else {
          timestamp = new Date(dateStr).getTime()
        }
      } catch {
        timestamp = minTime
      }

      // Normalize to 0-1 range
      const normalizedTime = (timestamp - minTime) / timeRange
      const elevation = normalizedTime * 1000000 // Scale for visibility

      return {
        position: [e.longitude!, e.latitude!],
        elevation,
        category: e.category || 'unknown',
        text: e.text,
        timestamp: dateStr,
        document_title: e.document_title || e.document_id,
      }
    })
  }, [spatioTemporalEvents])

  // Create Deck.gl layer
  const layers = useMemo(() => {
    return [
      new ColumnLayer({
        id: 'space-time-cube',
        data: processedEvents,
        diskResolution: 12,
        radius: radiusScale,
        elevationScale,
        extruded: true,
        pickable: true,
        getPosition: (d: any) => d.position,
        getElevation: (d: any) => d.elevation,
        getFillColor: (d: any) => {
          const color = CATEGORY_COLORS[d.category.toLowerCase()] || CATEGORY_COLORS['unknown']
          return [...color, 200]
        },
        getLineColor: [0, 0, 0],
        lineWidthMinPixels: 1,
        coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
      }),
    ]
  }, [processedEvents, elevationScale, radiusScale])

  // Tooltip
  const getTooltip = ({ object }: any) => {
    if (!object) return null
    return {
      html: `
        <div style="font-family: sans-serif; padding: 8px; background: white; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);">
          <div style="font-weight: bold; margin-bottom: 4px;">${object.text}</div>
          <div style="font-size: 12px; color: #666;">
            <div>Time: ${object.timestamp}</div>
            <div>Category: ${object.category}</div>
            <div>Document: ${object.document_title}</div>
          </div>
        </div>
      `,
      style: {
        backgroundColor: 'transparent',
        fontSize: '0.8em',
      },
    }
  }

  if (spatioTemporalEvents.length === 0) {
    return (
      <Box height={height} display="flex" alignItems="center" justifyContent="center">
        <Text color="gray.500" fontSize="lg">
          No spatiotemporal events available for 3D visualization
        </Text>
      </Box>
    )
  }

  // Don't render DeckGL on server side or before ready
  if (!isMounted || !deckGLReady) {
    return (
      <Box height={height} display="flex" alignItems="center" justifyContent="center">
        <Text color="gray.500" fontSize="lg">
          Loading 3D visualization...
        </Text>
      </Box>
    )
  }

  // Check WebGL support
  if (!webGLSupported) {
    return (
      <Box height={height} display="flex" alignItems="center" justifyContent="center">
        <VStack>
          <Text color="red.500" fontSize="lg">
            WebGL is not supported in your browser
          </Text>
          <Text color="gray.500" fontSize="sm">
            Please try a different browser or enable WebGL to view the 3D visualization
          </Text>
        </VStack>
      </Box>
    )
  }

  // Show initialization error if any
  if (initError) {
    return (
      <Box height={height} display="flex" alignItems="center" justifyContent="center">
        <VStack>
          <Text color="red.500" fontSize="lg">
            Failed to initialize 3D visualization
          </Text>
          <Text color="gray.500" fontSize="sm">
            {initError}
          </Text>
        </VStack>
      </Box>
    )
  }

  return (
    <VStack align="stretch" spacing={4}>
      {/* Controls */}
      <HStack spacing={6} flexWrap="wrap">
        <FormControl display="flex" alignItems="center" width="auto">
          <FormLabel mb={0} fontSize="sm" mr={2}>
            Labels
          </FormLabel>
          <Switch
            isChecked={showLabels}
            onChange={(e) => setShowLabels(e.target.checked)}
            size="sm"
          />
        </FormControl>

        <FormControl width="200px">
          <FormLabel fontSize="sm" mb={1}>
            Column Height: {elevationScale / 1000}k
          </FormLabel>
          <Slider
            value={elevationScale}
            min={10000}
            max={100000}
            step={10000}
            onChange={(val) => setElevationScale(val)}
            size="sm"
          >
            <SliderTrack>
              <SliderFilledTrack />
            </SliderTrack>
            <SliderThumb />
          </Slider>
        </FormControl>

        <FormControl width="200px">
          <FormLabel fontSize="sm" mb={1}>
            Column Radius: {radiusScale / 1000}k
          </FormLabel>
          <Slider
            value={radiusScale}
            min={5000}
            max={50000}
            step={5000}
            onChange={(val) => setRadiusScale(val)}
            size="sm"
          >
            <SliderTrack>
              <SliderFilledTrack />
            </SliderTrack>
            <SliderThumb />
          </Slider>
        </FormControl>
      </HStack>

      {/* Stats */}
      <HStack spacing={4} fontSize="sm" color="gray.600">
        <Text>
          <strong>{processedEvents.length}</strong> spatiotemporal events
        </Text>
        <Text>
          <strong>{new Set(processedEvents.map((e) => e.category)).size}</strong> categories
        </Text>
      </HStack>

      {/* 3D Visualization */}
      <Box
        ref={containerRef}
        height={height}
        position="relative"
        borderRadius="md"
        overflow="hidden"
        boxShadow="md"
      >
        {(() => {
          try {
            return (
              <DeckGL
                key={`deckgl-${deckGLKeyRef.current}`}
                viewState={viewState}
                onViewStateChange={({ viewState }: any) => setViewState(viewState)}
                controller={true}
                layers={layers}
                getTooltip={getTooltip}
                useDevicePixels={1}
                onError={(error: Error) => {
                  console.error('DeckGL error:', error)
                  setInitError(error.message)
                }}
                onWebGLInitialized={(gl: WebGLRenderingContext) => {
                  console.log('WebGL initialized successfully')
                }}
              >
                {/* Base map (optional) - using solid color background */}
                <div
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    background: '#f7fafc',
                    zIndex: -1,
                  }}
                />
              </DeckGL>
            )
          } catch (error: any) {
            console.error('Error rendering DeckGL:', error)
            setInitError(error.message || 'Unknown error')
            return (
              <Box
                display="flex"
                alignItems="center"
                justifyContent="center"
                height="100%"
              >
                <Text color="red.500">Failed to render 3D visualization</Text>
              </Box>
            )
          }
        })()}
      </Box>

      {/* Legend */}
      <HStack spacing={4} flexWrap="wrap" fontSize="xs">
        {Object.entries(CATEGORY_COLORS).map(([category, color]) => (
          <HStack key={category} spacing={1}>
            <Box
              width="12px"
              height="12px"
              bg={`rgb(${color[0]}, ${color[1]}, ${color[2]})`}
              borderRadius="sm"
            />
            <Text>{category}</Text>
          </HStack>
        ))}
      </HStack>

      {/* Instructions */}
      <Text fontSize="xs" color="gray.500" fontStyle="italic">
        💡 Drag to rotate • Scroll to zoom • Hold Shift + drag to pan • The height of each column represents time (earlier events at bottom, recent events at top)
      </Text>
    </VStack>
  )
}
