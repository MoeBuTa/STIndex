'use client'

import { useEffect, useRef, useMemo, useState } from 'react'
import * as d3 from 'd3'
import {
  Box,
  Text,
  VStack,
  HStack,
  Badge,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  useDisclosure,
  Divider,
} from '@chakra-ui/react'
import { SpatioTemporalEvent, BurstPeriod } from '../lib/analytics'

interface BackendBurstPeriod {
  start: string
  end: string
  event_count: number
  burst_intensity: number
}

interface StoryTimelineProps {
  events: SpatioTemporalEvent[]
  burstPeriods?: BackendBurstPeriod[]
  height?: number
  showBursts?: boolean
  width?: number  // Optional width for horizontal scrolling
}

export function StoryTimeline({
  events,
  burstPeriods = [],
  height = 400,
  showBursts = true,
  width = 1200,
}: StoryTimelineProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const [selectedEvent, setSelectedEvent] = useState<SpatioTemporalEvent | null>(null)
  const [isMounted, setIsMounted] = useState(false)
  const { isOpen, onOpen, onClose } = useDisclosure()

  // Ensure client-side only rendering for D3
  useEffect(() => {
    setIsMounted(true)
  }, [])

  // Filter events with temporal data
  const temporalEvents = useMemo(() => {
    return events.filter((e) => e.timestamp || e.normalized_date)
  }, [events])

  // Transform backend burst periods to component format
  const bursts = useMemo<BurstPeriod[]>(() => {
    if (!showBursts || !burstPeriods || burstPeriods.length === 0) return []

    return burstPeriods.map((bp, idx) => ({
      id: `burst-${idx}`,
      start: bp.start,
      end: bp.end,
      peakTime: bp.start, // Use start time as peak
      intensity: bp.burst_intensity,
      eventCount: bp.event_count,
      events: [], // Not needed for visualization
    }))
  }, [burstPeriods, showBursts])

  useEffect(() => {
    if (!isMounted || !svgRef.current || temporalEvents.length === 0) return

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove()

    // Group events by category for multi-track
    const parseDate = (dateStr: string | undefined) => {
      if (!dateStr) return null
      if (dateStr.startsWith('P')) return null
      try {
        return new Date(dateStr)
      } catch {
        return null
      }
    }

    const validEvents = temporalEvents
      .map((e) => ({
        ...e,
        date: parseDate(e.timestamp || e.normalized_date || ''),
      }))
      .filter((e) => e.date !== null) as Array<SpatioTemporalEvent & { date: Date }>

    if (validEvents.length === 0) return

    const categories = Array.from(new Set(validEvents.map((e) => e.category || 'unknown')))

    // Calculate dynamic dimensions
    const minTrackHeight = 40 // Minimum height per category
    const dynamicHeight = Math.max(height, categories.length * minTrackHeight + 100)

    const margin = {
      top: 40,
      right: 30,
      bottom: 60,
      left: 150  // Increased for longer category labels
    }

    const svgWidth = Math.max(width, 1200) // Ensure minimum width for scrolling
    const plotWidth = svgWidth - margin.left - margin.right
    const plotHeight = dynamicHeight - margin.top - margin.bottom

    const svg = d3
      .select(svgRef.current)
      .attr('width', svgWidth)
      .attr('height', dynamicHeight)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    const trackHeight = plotHeight / Math.max(categories.length, 1)

    // Create scales
    const xScale = d3
      .scaleTime()
      .domain(d3.extent(validEvents, (d) => d.date) as [Date, Date])
      .range([0, plotWidth])

    const categoryScale = d3
      .scaleBand()
      .domain(categories)
      .range([0, plotHeight])
      .padding(0.2)

    const colorScale = d3
      .scaleOrdinal<string>()
      .domain(categories)
      .range(d3.schemeCategory10)

    // Add x-axis
    const xAxis = d3.axisBottom(xScale).ticks(8)
    svg
      .append('g')
      .attr('transform', `translate(0,${plotHeight})`)
      .call(xAxis)
      .selectAll('text')
      .style('font-size', '11px')
      .attr('transform', 'rotate(-45)')
      .style('text-anchor', 'end')

    // Add y-axis (categories)
    const yAxis = d3.axisLeft(categoryScale)
    svg
      .append('g')
      .call(yAxis)
      .selectAll('text')
      .style('font-size', '11px')
      .style('text-overflow', 'ellipsis')
      .style('overflow', 'hidden')
      .style('white-space', 'nowrap')
      .style('max-width', `${margin.left - 20}px`)

    // Draw burst periods
    if (showBursts && bursts.length > 0) {
      bursts.forEach((burst) => {
        const startDate = parseDate(burst.start)
        const endDate = parseDate(burst.end)
        if (!startDate || !endDate) return

        svg
          .append('rect')
          .attr('x', xScale(startDate))
          .attr('y', 0)
          .attr('width', xScale(endDate) - xScale(startDate))
          .attr('height', plotHeight)
          .attr('fill', '#ff6b6b')
          .attr('opacity', 0.1)
          .attr('stroke', '#ff6b6b')
          .attr('stroke-width', 1)
          .attr('stroke-dasharray', '4,4')

        // Add burst label
        svg
          .append('text')
          .attr('x', (xScale(startDate) + xScale(endDate)) / 2)
          .attr('y', -10)
          .attr('text-anchor', 'middle')
          .attr('font-size', '10px')
          .attr('fill', '#ff6b6b')
          .text(`Burst: ${burst.eventCount} events`)
      })
    }

    // Draw events
    validEvents.forEach((event) => {
      const category = event.category || 'unknown'
      const yPos = (categoryScale(category) || 0) + (categoryScale.bandwidth() || 0) / 2

      const circle = svg
        .append('circle')
        .attr('cx', xScale(event.date))
        .attr('cy', yPos)
        .attr('r', 4)
        .attr('fill', colorScale(category))
        .attr('stroke', 'white')
        .attr('stroke-width', 1.5)
        .attr('opacity', 0.8)
        .style('cursor', 'pointer')

      // Tooltip and click handlers
      circle
        .on('mouseover', function (mouseEvent) {
          d3.select(this).attr('r', 6).attr('opacity', 1)

          if (tooltipRef.current) {
            const tooltip = d3.select(tooltipRef.current)
            tooltip
              .style('opacity', 1)
              .style('left', `${mouseEvent.pageX + 10}px`)
              .style('top', `${mouseEvent.pageY - 10}px`)
              .html(
                `
                <div>
                  <div style="font-weight: 600; margin-bottom: 6px; font-size: 14px;">${event.text}</div>
                  <div style="color: rgba(255,255,255,0.8); font-size: 12px; margin-bottom: 4px;">${event.date.toLocaleDateString()}</div>
                  ${event.category ? `<div style="color: #60a5fa; font-size: 12px; margin-bottom: 6px;">${event.category}</div>` : ''}
                  <div style="color: #60a5fa; font-size: 11px; margin-top: 8px; font-style: italic;">Click for details</div>
                </div>
              `
              )
          }
        })
        .on('mouseout', function () {
          d3.select(this).attr('r', 4).attr('opacity', 0.8)

          if (tooltipRef.current) {
            d3.select(tooltipRef.current).style('opacity', 0)
          }
        })
        .on('click', function () {
          // Hide tooltip when clicking to open modal
          if (tooltipRef.current) {
            d3.select(tooltipRef.current).style('opacity', 0)
          }
          // Reset circle size
          d3.select(this).attr('r', 4).attr('opacity', 0.8)

          setSelectedEvent(event)
          onOpen()
        })
    })

    // Add title
    svg
      .append('text')
      .attr('x', plotWidth / 2)
      .attr('y', -20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text('Multi-Track Event Timeline')

  }, [isMounted, temporalEvents, bursts, height, width, showBursts])

  if (!isMounted) {
    return (
      <Box p={6} bg="gray.50" borderRadius="md" height={`${height}px`}>
        <VStack justify="center" h="full">
          <Text color="gray.500">Loading timeline...</Text>
        </VStack>
      </Box>
    )
  }

  if (temporalEvents.length === 0) {
    return (
      <Box p={6} bg="gray.50" borderRadius="md" height={`${height}px`}>
        <VStack justify="center" h="full">
          <Text color="gray.500">No temporal data available for timeline</Text>
        </VStack>
      </Box>
    )
  }

  return (
    <Box position="relative">
      {/* Scrollable container for timeline */}
      <Box
        ref={containerRef}
        overflowX="auto"
        overflowY="auto"
        maxHeight={`${height + 100}px`}
        border="1px solid"
        borderColor="gray.200"
        borderRadius="md"
        bg="white"
        sx={{
          '&::-webkit-scrollbar': {
            width: '8px',
            height: '8px',
          },
          '&::-webkit-scrollbar-track': {
            bg: 'gray.100',
          },
          '&::-webkit-scrollbar-thumb': {
            bg: 'gray.400',
            borderRadius: '4px',
          },
          '&::-webkit-scrollbar-thumb:hover': {
            bg: 'gray.500',
          },
        }}
      >
        <svg ref={svgRef} style={{ display: 'block' }} />
      </Box>

      <div
        ref={tooltipRef}
        style={{
          position: 'fixed',
          opacity: 0,
          background: 'rgba(0, 0, 0, 0.9)',
          color: 'white',
          padding: '12px 16px',
          borderRadius: '8px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
          pointerEvents: 'none',
          zIndex: 9999,
          maxWidth: '300px',
          fontSize: '13px',
          lineHeight: '1.4',
          transition: 'opacity 0.15s ease-in-out',
        }}
      />

      {/* Event Details Modal */}
      <Modal isOpen={isOpen} onClose={onClose} size="lg">
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Event Details</ModalHeader>
          <ModalCloseButton />
          <ModalBody pb={6}>
            {selectedEvent && (
              <VStack align="stretch" spacing={4}>
                <Box>
                  <Text fontSize="sm" color="gray.500" fontWeight="medium" mb={1}>
                    Event Text
                  </Text>
                  <Text fontSize="lg" fontWeight="bold">
                    {selectedEvent.text}
                  </Text>
                </Box>

                <Divider />

                <HStack spacing={4} flexWrap="wrap">
                  {selectedEvent.category && (
                    <Box>
                      <Text fontSize="xs" color="gray.500" fontWeight="medium" mb={1}>
                        Category
                      </Text>
                      <Badge colorScheme="blue">{selectedEvent.category}</Badge>
                    </Box>
                  )}
                  {selectedEvent.confidence !== undefined && (
                    <Box>
                      <Text fontSize="xs" color="gray.500" fontWeight="medium" mb={1}>
                        Confidence
                      </Text>
                      <Badge colorScheme="green">
                        {Math.round(selectedEvent.confidence * 100)}%
                      </Badge>
                    </Box>
                  )}
                </HStack>

                <Divider />

                <VStack align="stretch" spacing={3}>
                  {selectedEvent.timestamp && (
                    <Box>
                      <Text fontSize="xs" color="gray.500" fontWeight="medium" mb={1}>
                        Timestamp
                      </Text>
                      <Text fontSize="sm">{selectedEvent.timestamp}</Text>
                    </Box>
                  )}

                  {selectedEvent.normalized_date && (
                    <Box>
                      <Text fontSize="xs" color="gray.500" fontWeight="medium" mb={1}>
                        Normalized Date
                      </Text>
                      <Text fontSize="sm">{selectedEvent.normalized_date}</Text>
                    </Box>
                  )}

                  {selectedEvent.location && (
                    <Box>
                      <Text fontSize="xs" color="gray.500" fontWeight="medium" mb={1}>
                        Location
                      </Text>
                      <Text fontSize="sm">{selectedEvent.location}</Text>
                    </Box>
                  )}

                  {selectedEvent.normalized_location && (
                    <Box>
                      <Text fontSize="xs" color="gray.500" fontWeight="medium" mb={1}>
                        Normalized Location
                      </Text>
                      <Text fontSize="sm">{selectedEvent.normalized_location}</Text>
                    </Box>
                  )}

                  {selectedEvent.source && (
                    <Box>
                      <Text fontSize="xs" color="gray.500" fontWeight="medium" mb={1}>
                        Source
                      </Text>
                      <Text fontSize="sm">{selectedEvent.source}</Text>
                    </Box>
                  )}

                  {selectedEvent.document_id && (
                    <Box>
                      <Text fontSize="xs" color="gray.500" fontWeight="medium" mb={1}>
                        Document ID
                      </Text>
                      <Text fontSize="xs" fontFamily="mono">
                        {selectedEvent.document_id}
                      </Text>
                    </Box>
                  )}

                  {selectedEvent.chunk_id && (
                    <Box>
                      <Text fontSize="xs" color="gray.500" fontWeight="medium" mb={1}>
                        Chunk ID
                      </Text>
                      <Text fontSize="xs" fontFamily="mono">
                        {selectedEvent.chunk_id}
                      </Text>
                    </Box>
                  )}
                </VStack>

                {selectedEvent.reflection_scores && (
                  <>
                    <Divider />
                    <Box>
                      <Text fontSize="sm" color="gray.500" fontWeight="medium" mb={2}>
                        Quality Scores
                      </Text>
                      <VStack align="stretch" spacing={2}>
                        <HStack justify="space-between">
                          <Text fontSize="sm">Relevance:</Text>
                          <Badge colorScheme="purple">
                            {selectedEvent.reflection_scores.relevance.toFixed(2)}
                          </Badge>
                        </HStack>
                        <HStack justify="space-between">
                          <Text fontSize="sm">Accuracy:</Text>
                          <Badge colorScheme="purple">
                            {selectedEvent.reflection_scores.accuracy.toFixed(2)}
                          </Badge>
                        </HStack>
                        <HStack justify="space-between">
                          <Text fontSize="sm">Completeness:</Text>
                          <Badge colorScheme="purple">
                            {selectedEvent.reflection_scores.completeness.toFixed(2)}
                          </Badge>
                        </HStack>
                        <HStack justify="space-between">
                          <Text fontSize="sm">Consistency:</Text>
                          <Badge colorScheme="purple">
                            {selectedEvent.reflection_scores.consistency.toFixed(2)}
                          </Badge>
                        </HStack>
                      </VStack>
                      {selectedEvent.reflection_scores.reasoning && (
                        <Box mt={3} p={3} bg="gray.50" borderRadius="md">
                          <Text fontSize="xs" color="gray.500" fontWeight="medium" mb={1}>
                            Reasoning:
                          </Text>
                          <Text fontSize="sm">{selectedEvent.reflection_scores.reasoning}</Text>
                        </Box>
                      )}
                    </Box>
                  </>
                )}
              </VStack>
            )}
          </ModalBody>
        </ModalContent>
      </Modal>

      {/* Statistics */}
      <Box mt={4}>
        <Badge colorScheme="blue">{temporalEvents.length} events</Badge>
      </Box>
    </Box>
  )
}
