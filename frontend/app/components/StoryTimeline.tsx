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

interface BackendStoryArc {
  story_id: string
  length: number
  confidence: number
  temporal_span: {
    start: string
    end: string
    duration_days: number
  }
}

interface BackendBurstPeriod {
  start_time: string
  end_time: string
  event_count: number
  intensity: number
}

interface StoryTimelineProps {
  events: SpatioTemporalEvent[]
  storyArcs: BackendStoryArc[]
  burstPeriods?: BackendBurstPeriod[]
  height?: number
  showBursts?: boolean
  showStoryArcs?: boolean
}

export function StoryTimeline({
  events,
  storyArcs,
  burstPeriods = [],
  height = 400,
  showBursts = true,
  showStoryArcs = true,
}: StoryTimelineProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const [selectedEvent, setSelectedEvent] = useState<SpatioTemporalEvent | null>(null)
  const { isOpen, onOpen, onClose } = useDisclosure()

  // Filter events with temporal data
  const temporalEvents = useMemo(() => {
    return events.filter((e) => e.timestamp || e.normalized_date)
  }, [events])

  // Transform backend burst periods to component format
  const bursts = useMemo<BurstPeriod[]>(() => {
    if (!showBursts || !burstPeriods || burstPeriods.length === 0) return []

    return burstPeriods.map((bp, idx) => ({
      id: `burst-${idx}`,
      start: bp.start_time,
      end: bp.end_time,
      peakTime: bp.start_time, // Use start time as peak
      intensity: bp.intensity,
      eventCount: bp.event_count,
      events: [], // Not needed for visualization
    }))
  }, [burstPeriods, showBursts])

  useEffect(() => {
    if (!svgRef.current || temporalEvents.length === 0) return

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove()

    const margin = {
      top: 40,
      right: 30,
      bottom: showStoryArcs && storyArcs.length > 0 ? 80 + (storyArcs.length * 15) : 60,
      left: 60
    }
    const width = svgRef.current.clientWidth - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    const svg = d3
      .select(svgRef.current)
      .attr('width', width + margin.left + margin.right)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Parse dates and prepare data
    const parseDate = (dateStr: string | undefined) => {
      if (!dateStr) return null
      // Handle ISO 8601 durations
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

    // Group events by category for multi-track
    const categories = Array.from(new Set(validEvents.map((e) => e.category || 'unknown')))
    const trackHeight = innerHeight / Math.max(categories.length, 1)

    // Create scales
    const xScale = d3
      .scaleTime()
      .domain(d3.extent(validEvents, (d) => d.date) as [Date, Date])
      .range([0, width])

    const categoryScale = d3
      .scaleBand()
      .domain(categories)
      .range([0, innerHeight])
      .padding(0.2)

    const colorScale = d3
      .scaleOrdinal<string>()
      .domain(categories)
      .range(d3.schemeCategory10)

    // Add x-axis
    const xAxis = d3.axisBottom(xScale).ticks(6)
    svg
      .append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis)
      .selectAll('text')
      .style('font-size', '12px')

    // Add y-axis (categories)
    const yAxis = d3.axisLeft(categoryScale)
    svg.append('g').call(yAxis).selectAll('text').style('font-size', '12px')

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
          .attr('height', innerHeight)
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

    // Draw story arcs
    if (showStoryArcs && storyArcs.length > 0) {
      storyArcs.forEach((story, idx) => {
        if (!story.temporal_span?.start || !story.temporal_span?.end) return

        const storyPoints = [
          new Date(story.temporal_span.start),
          new Date(story.temporal_span.end)
        ].filter((d) => d && !isNaN(d.getTime()))

        if (storyPoints.length < 2) return

        const startX = xScale(storyPoints[0])
        const endX = xScale(storyPoints[1])
        const arcY = innerHeight + 20 + (idx * 15) // Position below the timeline

        // Draw arc line
        svg
          .append('line')
          .attr('x1', startX)
          .attr('y1', arcY)
          .attr('x2', endX)
          .attr('y2', arcY)
          .attr('stroke', '#3182ce')
          .attr('stroke-width', 3)
          .attr('opacity', 0.7)

        // Draw start marker
        svg
          .append('circle')
          .attr('cx', startX)
          .attr('cy', arcY)
          .attr('r', 5)
          .attr('fill', '#3182ce')

        // Draw end marker
        svg
          .append('circle')
          .attr('cx', endX)
          .attr('cy', arcY)
          .attr('r', 5)
          .attr('fill', '#3182ce')

        // Add story label with details
        const labelText = `Story ${idx + 1} (${story.length} clusters, ${story.temporal_span.duration_days}d, ${(story.confidence * 100).toFixed(0)}%)`
        svg
          .append('text')
          .attr('x', startX)
          .attr('y', arcY - 8)
          .attr('font-size', '10px')
          .attr('fill', '#3182ce')
          .attr('font-weight', 'bold')
          .text(labelText)
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
                <div style="font-size: 12px;">
                  <strong>${event.text}</strong><br/>
                  <span style="color: #666;">${event.date.toLocaleDateString()}</span><br/>
                  ${event.category ? `<span style="color: ${colorScale(event.category)};">${event.category}</span><br/>` : ''}
                  <span style="color: #3182ce; font-size: 10px;">Click for details</span>
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
          setSelectedEvent(event)
          onOpen()
        })
    })

    // Add title
    svg
      .append('text')
      .attr('x', width / 2)
      .attr('y', -20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text('Temporal Event Timeline')

  }, [temporalEvents, bursts, storyArcs, height, showBursts, showStoryArcs])

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
      <svg ref={svgRef} style={{ width: '100%', height: `${height}px` }} />
      <div
        ref={tooltipRef}
        style={{
          position: 'absolute',
          opacity: 0,
          background: 'white',
          padding: '8px',
          borderRadius: '4px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
          pointerEvents: 'none',
          zIndex: 1000,
          transition: 'opacity 0.2s',
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

      {/* Legend */}
      <Box mt={4}>
        <HStack spacing={4} flexWrap="wrap">
          {showBursts && bursts.length > 0 && (
            <HStack spacing={2}>
              <Box w="20px" h="10px" bg="red.200" border="1px dashed" borderColor="red.500" />
              <Text fontSize="xs">Burst Period</Text>
            </HStack>
          )}
          {showStoryArcs && storyArcs.length > 0 && (
            <HStack spacing={2}>
              <Box w="20px" h="3px" bg="blue.500" />
              <Text fontSize="xs">Story Arc (progression across time)</Text>
            </HStack>
          )}
        </HStack>
      </Box>

      {/* Statistics */}
      <Box mt={4}>
        <VStack align="start" spacing={1}>
          <HStack spacing={4}>
            <Badge colorScheme="blue">{temporalEvents.length} events</Badge>
            {showBursts && <Badge colorScheme="red">{bursts.length} bursts</Badge>}
            {showStoryArcs && <Badge colorScheme="purple">{storyArcs.length} stories</Badge>}
          </HStack>
        </VStack>
      </Box>
    </Box>
  )
}
