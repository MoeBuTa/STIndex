'use client'

import { SimpleGrid, Box, Text, Stat, StatLabel, StatNumber, StatHelpText } from '@chakra-ui/react'

interface ExtractionResult {
  chunk_id: string
  extraction: {
    success: boolean
    entities?: {
      temporal?: any[]
      spatial?: any[]
      [key: string]: any[] | undefined
    }
  }
}

interface DashboardStatsProps {
  data: ExtractionResult[]
}

export function DashboardStats({ data }: DashboardStatsProps) {
  // Calculate statistics
  const totalChunks = data.length

  // Count unique documents
  const uniqueDocs = new Set(data.map((item) => item.chunk_id.split('_chunk_')[0]))
  const totalDocuments = uniqueDocs.size

  // Count temporal entities
  const totalTemporal = data.reduce(
    (sum, item) => sum + (item.extraction.entities?.temporal?.length || 0),
    0
  )

  // Count spatial entities
  const totalSpatial = data.reduce(
    (sum, item) => sum + (item.extraction.entities?.spatial?.length || 0),
    0
  )

  // Count custom dimension entities (anything not temporal/spatial)
  const totalCustomDimensions = data.reduce((sum, item) => {
    let count = 0

    // Count from entities object
    if (item.extraction.entities) {
      const customKeys = Object.keys(item.extraction.entities).filter(
        (key) => key !== 'temporal' && key !== 'spatial'
      )
      count += customKeys.reduce((keySum, key) => {
        const entities = item.extraction.entities?.[key]
        return keySum + (entities?.length || 0)
      }, 0)
    }

    return sum + count
  }, 0)

  const stats = [
    { label: 'Documents', value: totalDocuments, help: `${totalChunks} chunks processed` },
    { label: 'Temporal Entities', value: totalTemporal, help: 'Dates, times, durations' },
    { label: 'Spatial Entities', value: totalSpatial, help: 'Locations, regions' },
    { label: 'Custom Dimensions', value: totalCustomDimensions, help: 'Event types, diseases, etc.' },
  ]

  return (
    <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={6}>
      {stats.map((stat, index) => (
        <Box key={index} bg="white" p={6} borderRadius="lg" shadow="md">
          <Stat>
            <StatLabel fontSize="sm" color="gray.600">{stat.label}</StatLabel>
            <StatNumber fontSize="3xl">{stat.value}</StatNumber>
            <StatHelpText fontSize="xs" color="gray.500">{stat.help}</StatHelpText>
          </Stat>
        </Box>
      ))}
    </SimpleGrid>
  )
}
