'use client'

import { useMemo } from 'react'
import {
  Box,
  SimpleGrid,
  VStack,
  HStack,
  Text,
  Badge,
  Progress,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Divider,
  List,
  ListItem,
  ListIcon,
} from '@chakra-ui/react'
import { MdTrendingUp, MdLocationOn, MdEvent, MdTimeline } from 'react-icons/md'
import { SpatioTemporalEvent } from '../lib/analytics'

interface BackendClusters {
  clusters: any[]
  burst_periods: any[]
  statistics: any
}

interface AnalyticsPanelsProps {
  events: SpatioTemporalEvent[]
  backendClusters: BackendClusters | null
}

export function AnalyticsPanels({ events, backendClusters }: AnalyticsPanelsProps) {
  // Helper function to parse various date formats
  const parseNormalizedDate = (dateStr: string | undefined) => {
    if (!dateStr) return null

    // Skip durations (P9Y, P7D, etc.)
    if (dateStr.startsWith('P')) return null

    // Handle intervals (2023-01-01/2023-12-31)
    if (dateStr.includes('/')) {
      const [start, end] = dateStr.split('/')
      return { start: new Date(start), end: new Date(end), isInterval: true }
    }

    // Handle year only (2023)
    if (/^\d{4}$/.test(dateStr)) {
      return new Date(dateStr + '-01-01')
    }

    // Handle year-month (2020-08)
    if (/^\d{4}-\d{2}$/.test(dateStr)) {
      return new Date(dateStr + '-01')
    }

    // Handle full date
    return new Date(dateStr)
  }

  // Check if a date falls within a burst period
  const isInBurst = (normalizedDate: string | undefined, burstStart: string, burstEnd: string) => {
    const parsedDate = parseNormalizedDate(normalizedDate)
    if (!parsedDate) return false

    const start = new Date(burstStart)
    const end = new Date(burstEnd)

    if ((parsedDate as any).isInterval) {
      // Check if interval overlaps with burst
      const interval = parsedDate as any
      return interval.start <= end && interval.end >= start
    }

    return parsedDate >= start && parsedDate <= end
  }

  // Compute analytics with safety checks
  const { bursts, qualityMetrics, dimensionStats } = useMemo(() => {
    // Safety check - return empty state if no events
    if (!events || events.length === 0) {
      return {
        bursts: [],
        qualityMetrics: {
          relevance: 0,
          accuracy: 0,
          completeness: 0,
          consistency: 0,
          totalEvents: 0,
          withScores: 0,
          avgConfidence: 0,
        },
        dimensionStats: {
          totalEvents: 0,
          temporalEvents: 0,
          spatialEvents: 0,
          categories: [],
          topSources: [],
          temporalCoverage: '0',
          spatialCoverage: '0',
        },
      }
    }

    const temporalEvents = events.filter((e) => e.timestamp || e.normalized_date)
    const spatialEvents = events.filter((e) => e.latitude && e.longitude)

    // Use backend-provided burst periods and map to expected format
    const burstsData = (backendClusters?.burst_periods || []).map((bp: any) => {
      // Filter events within this burst period using proper date parsing
      const burstEvents = events.filter((e) => {
        const eventDate = e.timestamp || e.normalized_date
        if (!eventDate) return false
        return isInBurst(eventDate, bp.start, bp.end)
      })

      // Calculate dominant location (using event text as location identifier)
      const locationCount = new Map<string, number>()
      burstEvents.forEach((e) => {
        if (e.text) {
          locationCount.set(e.text, (locationCount.get(e.text) || 0) + 1)
        }
      })
      const dominantLocation = Array.from(locationCount.entries())
        .sort((a, b) => b[1] - a[1])[0]?.[0]

      // Calculate dominant category
      const categoryCount = new Map<string, number>()
      burstEvents.forEach((e) => {
        if (e.category) {
          categoryCount.set(e.category, (categoryCount.get(e.category) || 0) + 1)
        }
      })
      const dominantCategory = Array.from(categoryCount.entries())
        .sort((a, b) => b[1] - a[1])[0]?.[0]

      return {
        start: bp.start,
        end: bp.end,
        eventCount: burstEvents.length, // Use frontend calculated count for display
        backendEventCount: bp.event_count, // Keep backend count for reference
        intensity: bp.burst_intensity,
        peakTime: bp.start, // Use start time as peak
        dominantLocation,
        dominantCategory,
      }
    })

    // Calculate quality metrics from reflection scores
    const eventsWithScores = events.filter(
      (e) => e.custom_dimensions && 'reflection_scores' in e.custom_dimensions
    )

    let avgRelevance = 0
    let avgAccuracy = 0
    let avgCompleteness = 0
    let avgConsistency = 0

    if (eventsWithScores.length > 0) {
      eventsWithScores.forEach((e: any) => {
        const scores = e.custom_dimensions?.reflection_scores
        if (scores) {
          avgRelevance += scores.relevance || 0
          avgAccuracy += scores.accuracy || 0
          avgCompleteness += scores.completeness || 0
          avgConsistency += scores.consistency || 0
        }
      })

      const count = eventsWithScores.length
      avgRelevance /= count
      avgAccuracy /= count
      avgCompleteness /= count
      avgConsistency /= count
    }

    const quality = {
      relevance: avgRelevance,
      accuracy: avgAccuracy,
      completeness: avgCompleteness,
      consistency: avgConsistency,
      totalEvents: events.length,
      withScores: eventsWithScores.length,
      avgConfidence:
        events.reduce((sum, e: any) => sum + (e.confidence || 0), 0) / events.length || 0,
    }

    // Dimension statistics
    const categories = new Map<string, number>()
    events.forEach((e) => {
      const cat = e.category || 'unknown'
      categories.set(cat, (categories.get(cat) || 0) + 1)
    })

    const sources = new Map<string, number>()
    events.forEach((e) => {
      sources.set(e.source, (sources.get(e.source) || 0) + 1)
    })

    const dimStats = {
      totalEvents: events.length,
      temporalEvents: temporalEvents.length,
      spatialEvents: spatialEvents.length,
      categories: Array.from(categories.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5),
      topSources: Array.from(sources.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3),
      temporalCoverage:
        temporalEvents.length > 0
          ? ((temporalEvents.length / events.length) * 100).toFixed(1)
          : '0',
      spatialCoverage:
        spatialEvents.length > 0
          ? ((spatialEvents.length / events.length) * 100).toFixed(1)
          : '0',
    }

    return {
      bursts: burstsData,
      qualityMetrics: quality,
      dimensionStats: dimStats,
    }
  }, [events, backendClusters])

  // Format date consistently for SSR/client (ISO format)
  const formatDateConsistent = (date: Date): string => {
    const year = date.getFullYear()
    const month = String(date.getMonth() + 1).padStart(2, '0')
    const day = String(date.getDate()).padStart(2, '0')
    return `${year}-${month}-${day}`
  }

  // Calculate temporal range
  const temporalRange = useMemo(() => {
    const temporalEvents = events.filter((e) => e.timestamp || e.normalized_date)
    if (temporalEvents.length === 0) return null

    const dates = temporalEvents
      .map((e) => parseNormalizedDate(e.timestamp || e.normalized_date))
      .filter((d) => d && !(d as any).isInterval)
      .sort((a: any, b: any) => a - b)

    if (dates.length === 0) return null

    const earliest = dates[0] as Date
    const latest = dates[dates.length - 1] as Date

    return {
      earliest: formatDateConsistent(earliest),
      latest: formatDateConsistent(latest),
      span: Math.ceil((latest.getTime() - earliest.getTime()) / (1000 * 60 * 60 * 24)),
    }
  }, [events])

  // Calculate category distribution percentages
  const categoryDistribution = useMemo(() => {
    const categories = new Map<string, number>()
    events.forEach((e) => {
      const cat = e.category || 'unknown'
      categories.set(cat, (categories.get(cat) || 0) + 1)
    })

    return Array.from(categories.entries())
      .map(([name, count]) => ({
        name,
        count,
        percentage: events.length > 0 ? (count / events.length) * 100 : 0,
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 8)
  }, [events])

  // Calculate confidence distribution
  const confidenceDistribution = useMemo(() => {
    const high = events.filter((e) => (e.confidence || 0) >= 0.8).length
    const medium = events.filter((e) => (e.confidence || 0) >= 0.5 && (e.confidence || 0) < 0.8)
      .length
    const low = events.filter((e) => (e.confidence || 0) < 0.5).length

    return { high, medium, low }
  }, [events])

  // Calculate geographic spread
  const geographicSpread = useMemo(() => {
    const uniqueLocations = new Set<string>()
    const locationTypes = new Map<string, number>()

    events.forEach((e) => {
      // Count unique locations from spatial events
      if (e.latitude && e.longitude && e.text) {
        uniqueLocations.add(e.text)
      }

      // Count location types
      if (e.custom_dimensions) {
        const locType = (e.custom_dimensions as any).location_type
        if (locType) {
          locationTypes.set(locType, (locationTypes.get(locType) || 0) + 1)
        }
      }
    })

    return {
      uniqueLocations: uniqueLocations.size,
      locationTypes: Array.from(locationTypes.entries()).sort((a, b) => b[1] - a[1]),
      locationsWithCoords: events.filter((e) => e.latitude && e.longitude).length,
    }
  }, [events])

  // Calculate burst coverage
  const burstCoverage = useMemo(() => {
    if (!backendClusters?.burst_periods || bursts.length === 0) return 0

    const eventsInBursts = events.filter((e) => {
      const eventDate = e.timestamp || e.normalized_date
      if (!eventDate) return false

      return bursts.some((burst) => isInBurst(eventDate, burst.start, burst.end))
    }).length

    return events.length > 0 ? (eventsInBursts / events.length) * 100 : 0
  }, [events, bursts, backendClusters])

  return (
    <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={6}>
      {/* Row 1, Col 1: Extraction Quality Metrics */}
      <Box bg="white" p={6} borderRadius="lg" shadow="md">
        <VStack align="stretch" spacing={4}>
          <HStack>
            <MdTrendingUp size={24} color="#3182ce" />
            <Text fontSize="lg" fontWeight="bold">
              Extraction Quality
            </Text>
            <Badge colorScheme="blue" fontSize="xs">
              {qualityMetrics.withScores} scored
            </Badge>
          </HStack>
          <Divider />

          <SimpleGrid columns={2} spacing={4}>
            <Stat>
              <StatLabel fontSize="xs">Relevance</StatLabel>
              <StatNumber fontSize="2xl">
                {(qualityMetrics.relevance * 100).toFixed(0)}%
              </StatNumber>
              <Progress
                value={qualityMetrics.relevance * 100}
                colorScheme="green"
                size="sm"
                mt={2}
              />
            </Stat>

            <Stat>
              <StatLabel fontSize="xs">Accuracy</StatLabel>
              <StatNumber fontSize="2xl">
                {(qualityMetrics.accuracy * 100).toFixed(0)}%
              </StatNumber>
              <Progress
                value={qualityMetrics.accuracy * 100}
                colorScheme="blue"
                size="sm"
                mt={2}
              />
            </Stat>

            <Stat>
              <StatLabel fontSize="xs">Completeness</StatLabel>
              <StatNumber fontSize="2xl">
                {(qualityMetrics.completeness * 100).toFixed(0)}%
              </StatNumber>
              <Progress
                value={qualityMetrics.completeness * 100}
                colorScheme="purple"
                size="sm"
                mt={2}
              />
            </Stat>

            <Stat>
              <StatLabel fontSize="xs">Consistency</StatLabel>
              <StatNumber fontSize="2xl">
                {(qualityMetrics.consistency * 100).toFixed(0)}%
              </StatNumber>
              <Progress
                value={qualityMetrics.consistency * 100}
                colorScheme="orange"
                size="sm"
                mt={2}
              />
            </Stat>
          </SimpleGrid>

          <Divider />

          <Box>
            <Text fontSize="xs" fontWeight="medium" mb={2}>
              Confidence Distribution
            </Text>
            <VStack spacing={2} align="stretch">
              <Box>
                <HStack justify="space-between" mb={1}>
                  <Text fontSize="xs" color="gray.600">
                    High ≥80%
                  </Text>
                  <Text fontSize="xs" fontWeight="bold">
                    {confidenceDistribution.high}
                  </Text>
                </HStack>
                <Progress
                  value={
                    events.length > 0
                      ? (confidenceDistribution.high / events.length) * 100
                      : 0
                  }
                  colorScheme="green"
                  size="sm"
                />
              </Box>
              <Box>
                <HStack justify="space-between" mb={1}>
                  <Text fontSize="xs" color="gray.600">
                    Medium 50-80%
                  </Text>
                  <Text fontSize="xs" fontWeight="bold">
                    {confidenceDistribution.medium}
                  </Text>
                </HStack>
                <Progress
                  value={
                    events.length > 0
                      ? (confidenceDistribution.medium / events.length) * 100
                      : 0
                  }
                  colorScheme="yellow"
                  size="sm"
                />
              </Box>
              <Box>
                <HStack justify="space-between" mb={1}>
                  <Text fontSize="xs" color="gray.600">
                    Low &lt;50%
                  </Text>
                  <Text fontSize="xs" fontWeight="bold">
                    {confidenceDistribution.low}
                  </Text>
                </HStack>
                <Progress
                  value={
                    events.length > 0 ? (confidenceDistribution.low / events.length) * 100 : 0
                  }
                  colorScheme="red"
                  size="sm"
                />
              </Box>
            </VStack>
          </Box>
        </VStack>
      </Box>

      {/* Row 1, Col 2: Event Burst Detection */}
      <Box bg="white" p={6} borderRadius="lg" shadow="md">
        <VStack align="stretch" spacing={4}>
          <HStack>
            <MdEvent size={24} color="#e53e3e" />
            <Text fontSize="lg" fontWeight="bold">
              Event Bursts
            </Text>
            <Badge colorScheme="red">
              {backendClusters?.burst_periods.length || bursts.length}
            </Badge>
          </HStack>
          <Divider />

          {(backendClusters?.burst_periods.length || bursts.length) === 0 ? (
            <Text fontSize="sm" color="gray.500">
              No significant burst periods detected
            </Text>
          ) : (
            <>
              <Box>
                <Text fontSize="xs" fontWeight="medium" mb={2}>
                  Burst Coverage
                </Text>
                <HStack justify="space-between" mb={1}>
                  <Text fontSize="xs" color="gray.600">
                    Events in bursts
                  </Text>
                  <Text fontSize="sm" fontWeight="bold">
                    {burstCoverage.toFixed(1)}%
                  </Text>
                </HStack>
                <Progress value={burstCoverage} colorScheme="red" size="sm" />
              </Box>

              <Divider />

              <VStack align="stretch" spacing={3} maxH="220px" overflowY="auto">
                {bursts.map((burst, idx) => (
                  <Box
                    key={idx}
                    p={3}
                    bg="red.50"
                    borderRadius="md"
                    borderLeft="3px solid"
                    borderColor="red.500"
                  >
                    <HStack justify="space-between" mb={1}>
                      <Text fontSize="sm" fontWeight="bold">
                        Burst {idx + 1}
                      </Text>
                      <Badge colorScheme="red" fontSize="xs">
                        {burst.eventCount} events
                      </Badge>
                    </HStack>
                    <Text fontSize="xs" color="gray.600">
                      {burst.start} → {burst.end}
                    </Text>
                    <Text fontSize="xs" color="gray.600">
                      Intensity: {burst.intensity?.toFixed(1) || 'N/A'}
                    </Text>
                    {burst.dominantCategory && (
                      <Badge colorScheme="purple" fontSize="xs" mt={1}>
                        {burst.dominantCategory}
                      </Badge>
                    )}
                  </Box>
                ))}
              </VStack>
            </>
          )}
        </VStack>
      </Box>

      {/* Row 2, Col 1: Temporal Analytics */}
      <Box bg="white" p={6} borderRadius="lg" shadow="md">
        <VStack align="stretch" spacing={4}>
          <HStack>
            <MdTimeline size={24} color="#805ad5" />
            <Text fontSize="lg" fontWeight="bold">
              Temporal Analytics
            </Text>
          </HStack>
          <Divider />

          {temporalRange ? (
            <>
              <SimpleGrid columns={2} spacing={4}>
                <Stat>
                  <StatLabel fontSize="xs">Earliest Event</StatLabel>
                  <StatNumber fontSize="lg">{temporalRange.earliest}</StatNumber>
                </Stat>

                <Stat>
                  <StatLabel fontSize="xs">Latest Event</StatLabel>
                  <StatNumber fontSize="lg">{temporalRange.latest}</StatNumber>
                </Stat>
              </SimpleGrid>

              <Box>
                <Text fontSize="xs" fontWeight="medium" mb={2}>
                  Time Span
                </Text>
                <HStack justify="space-between" mb={1}>
                  <Text fontSize="xs" color="gray.600">
                    {temporalRange.span} days
                  </Text>
                  <Text fontSize="xs" color="gray.600">
                    {(temporalRange.span / 365).toFixed(1)} years
                  </Text>
                </HStack>
                <Progress
                  value={Math.min((temporalRange.span / 365) * 10, 100)}
                  colorScheme="purple"
                  size="sm"
                />
              </Box>

              <Divider />

              <Box>
                <Text fontSize="xs" fontWeight="medium" mb={2}>
                  Cluster Distribution
                </Text>
                <HStack justify="space-between">
                  <Stat>
                    <StatLabel fontSize="xs">Total Clusters</StatLabel>
                    <StatNumber fontSize="xl">
                      {backendClusters?.clusters.length || 0}
                    </StatNumber>
                  </Stat>
                  <Stat>
                    <StatLabel fontSize="xs">Avg Events/Cluster</StatLabel>
                    <StatNumber fontSize="xl">
                      {backendClusters?.clusters.length
                        ? (
                            events.length / backendClusters.clusters.length
                          ).toFixed(1)
                        : 0}
                    </StatNumber>
                  </Stat>
                </HStack>
              </Box>
            </>
          ) : (
            <Text fontSize="sm" color="gray.500">
              No temporal data available
            </Text>
          )}
        </VStack>
      </Box>

      {/* Row 2, Col 2: Spatial Analytics */}
      <Box bg="white" p={6} borderRadius="lg" shadow="md">
        <VStack align="stretch" spacing={4}>
          <HStack>
            <MdLocationOn size={24} color="#38a169" />
            <Text fontSize="lg" fontWeight="bold">
              Spatial Analytics
            </Text>
          </HStack>
          <Divider />

          <SimpleGrid columns={2} spacing={4}>
            <Stat>
              <StatLabel fontSize="xs">Unique Locations</StatLabel>
              <StatNumber fontSize="2xl">{geographicSpread.uniqueLocations}</StatNumber>
              <StatHelpText fontSize="xs">Distinct places</StatHelpText>
            </Stat>

            <Stat>
              <StatLabel fontSize="xs">Geocoded</StatLabel>
              <StatNumber fontSize="2xl">{geographicSpread.locationsWithCoords}</StatNumber>
              <StatHelpText fontSize="xs">{dimensionStats.spatialCoverage}% coverage</StatHelpText>
            </Stat>
          </SimpleGrid>

          <Box>
            <Text fontSize="xs" fontWeight="medium" mb={2}>
              Location Types
            </Text>
            <VStack spacing={1} align="stretch">
              {geographicSpread.locationTypes.slice(0, 4).map(([type, count]) => (
                <HStack key={type} justify="space-between">
                  <Text fontSize="xs" textTransform="capitalize">
                    {type}
                  </Text>
                  <Badge colorScheme="green" fontSize="xs">
                    {count}
                  </Badge>
                </HStack>
              ))}
            </VStack>
          </Box>

          <Divider />

          <Box>
            <Text fontSize="xs" fontWeight="medium" mb={2}>
              Top Sources
            </Text>
            <List spacing={1}>
              {dimensionStats.topSources.slice(0, 3).map(([source, count]) => (
                <ListItem key={source} fontSize="xs">
                  <HStack justify="space-between">
                    <Text isTruncated maxW="200px">
                      {source}
                    </Text>
                    <Badge colorScheme="teal" fontSize="xs">
                      {count}
                    </Badge>
                  </HStack>
                </ListItem>
              ))}
            </List>
          </Box>
        </VStack>
      </Box>


    </SimpleGrid>
  )
}
