'use client'

import { useEffect, useRef, useMemo } from 'react'
import * as d3 from 'd3'
import { Box, Text, VStack, HStack, Badge } from '@chakra-ui/react'
import { detectBursts, extractStoryArcs, clusterEvents, SpatioTemporalEvent, BurstPeriod, StoryArc } from '../lib/analytics'

interface StoryTimelineProps {
  events: SpatioTemporalEvent[]
  height?: number
  showBursts?: boolean
  showStoryArcs?: boolean
}

export function StoryTimeline({
  events,
  height = 400,
  showBursts = true,
  showStoryArcs = true,
}: StoryTimelineProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)

  // Filter events with temporal data
  const temporalEvents = useMemo(() => {
    return events.filter((e) => e.timestamp || e.normalized_date)
  }, [events])

  // Detect bursts
  const bursts = useMemo(() => {
    if (!showBursts || temporalEvents.length === 0) return []
    return detectBursts(temporalEvents, 1, 3)
  }, [temporalEvents, showBursts])

  // Get story arcs
  const stories = useMemo(() => {
    if (!showStoryArcs || temporalEvents.length === 0) return []
    const clusters = clusterEvents(temporalEvents, 50, 7, 3)
    return extractStoryArcs(clusters, 0.3)
  }, [temporalEvents, showStoryArcs])

  useEffect(() => {
    if (!svgRef.current || temporalEvents.length === 0) return

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove()

    const margin = { top: 40, right: 30, bottom: 60, left: 60 }
    const width = svgRef.current.clientWidth - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    const svg = d3
      .select(svgRef.current)
      .attr('width', width + margin.left + margin.right)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Parse dates and prepare data
    const parseDate = (dateStr: string) => {
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
    if (showStoryArcs && stories.length > 0) {
      stories.forEach((story, idx) => {
        const storyPoints = story.timeline
          .map((timeStr) => parseDate(timeStr))
          .filter((d) => d !== null) as Date[]

        if (storyPoints.length < 2) return

        // Create arc path
        const arcGenerator = d3
          .line<Date>()
          .x((d) => xScale(d))
          .y((_, i) => innerHeight / 2 + Math.sin((i / storyPoints.length) * Math.PI) * 20)
          .curve(d3.curveBasis)

        svg
          .append('path')
          .datum(storyPoints)
          .attr('fill', 'none')
          .attr('stroke', '#3182ce')
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '5,5')
          .attr('opacity', 0.5)
          .attr('d', arcGenerator)

        // Add story label
        if (storyPoints.length > 0) {
          svg
            .append('text')
            .attr('x', xScale(storyPoints[0]))
            .attr('y', innerHeight + 35)
            .attr('font-size', '9px')
            .attr('fill', '#3182ce')
            .text(`Story ${idx + 1}`)
        }
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

      // Tooltip
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
                  ${event.source ? `<em>${event.source}</em>` : ''}
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

  }, [temporalEvents, bursts, stories, height, showBursts, showStoryArcs])

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

      {/* Legend */}
      <Box mt={4}>
        <HStack spacing={4} flexWrap="wrap">
          {showBursts && bursts.length > 0 && (
            <HStack spacing={2}>
              <Box w="20px" h="10px" bg="red.200" border="1px dashed" borderColor="red.500" />
              <Text fontSize="xs">Burst Period</Text>
            </HStack>
          )}
          {showStoryArcs && stories.length > 0 && (
            <HStack spacing={2}>
              <Box w="20px" h="2px" bg="blue.500" />
              <Text fontSize="xs">Story Arc</Text>
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
            {showStoryArcs && <Badge colorScheme="purple">{stories.length} stories</Badge>}
          </HStack>
        </VStack>
      </Box>
    </Box>
  )
}
