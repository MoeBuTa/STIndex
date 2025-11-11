'use client'

import { useEffect, useState } from 'react'
import { Box, Container, Heading, Text, Spinner, Center, VStack } from '@chakra-ui/react'
import { DashboardStats } from './components/DashboardStats'
import { TemporalTimeline } from './components/TemporalTimeline'
import { DimensionBreakdown } from './components/DimensionBreakdown'

interface ExtractionResult {
  chunk_id: string
  chunk_index: number
  document_id: string
  document_title: string
  extraction: {
    entities?: {
      temporal?: any[]
      spatial?: any[]
      [key: string]: any[] | undefined
    }
    temporal_entities: any[]
    spatial_entities: any[]
    event_type?: any[]
    disease?: any[]
    venue_type?: any[]
    success: boolean
    error?: string
    document_metadata: {
      source: string
      category: string
      topic: string
      jurisdiction: string
      year: number
    }
    extraction_config?: {
      enabled_dimensions?: string[]
      dimension_config_path?: string
    }
    dimension_configs?: Record<string, any>
  }
}

export default function Home() {
  const [data, setData] = useState<ExtractionResult[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/data/extraction_results.json')
      .then((res) => res.json())
      .then((results) => {
        setData(results)
        setLoading(false)
      })
      .catch((err) => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

  if (loading) {
    return (
      <Center minH="100vh" bg="gray.50">
        <VStack spacing={4}>
          <Spinner size="xl" color="blue.500" thickness="4px" />
          <Text color="gray.600">Loading extraction data...</Text>
        </VStack>
      </Center>
    )
  }

  if (error) {
    return (
      <Center minH="100vh" bg="gray.50">
        <Text color="red.500" fontSize="xl">Error loading data: {error}</Text>
      </Center>
    )
  }

  // Filter successful extractions
  const successfulExtractions = data.filter((item) => item.extraction.success)

  return (
    <Box minH="100vh" bg="gray.50" py={8}>
      <Container maxW="7xl" px={4}>
        <VStack spacing={8} align="stretch">
          <Box>
            <Heading as="h1" size="2xl" mb={2}>STIndex Dashboard</Heading>
            <Text color="gray.600" fontSize="lg">
              Multi-Dimensional Data Visualization
            </Text>
          </Box>

          <DashboardStats data={successfulExtractions} />

          <Box bg="white" p={6} borderRadius="lg" shadow="md">
            <Heading as="h2" size="xl" mb={4}>Temporal Timeline</Heading>
            <TemporalTimeline data={successfulExtractions} />
          </Box>

          <Box bg="white" p={6} borderRadius="lg" shadow="md">
            <Heading as="h2" size="xl" mb={4}>Dimension Analysis</Heading>
            <DimensionBreakdown data={successfulExtractions} />
          </Box>
        </VStack>
      </Container>
    </Box>
  )
}
