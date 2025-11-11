'use client'

import { useMemo } from 'react'
import { Box, Text, VStack, HStack, Badge, Flex } from '@chakra-ui/react'

interface TemporalEntity {
  text: string
  normalized?: string
  confidence?: number
  normalization_type?: string
  dimension_name?: string
  reflection_scores?: {
    relevance: number
    accuracy: number
    completeness: number
    consistency: number
    reasoning: string
  }
}

interface ExtractionResult {
  chunk_id: string
  document_id: string
  document_title: string
  source: string | null
  extraction: {
    success: boolean
    entities?: {
      temporal?: TemporalEntity[]
      [key: string]: any[] | undefined
    }
  }
}

interface TemporalTimelineProps {
  data: ExtractionResult[]
}

export function TemporalTimeline({ data }: TemporalTimelineProps) {
  // Extract all temporal entities and group by year
  const temporalData = useMemo(() => {
    const entities: Array<TemporalEntity & { document_title: string; source: string }> = []

    data.forEach((item) => {
      const temporal_entities = item.extraction.entities?.temporal || []
      temporal_entities.forEach((entity) => {
        entities.push({
          ...entity,
          document_title: item.document_title,
          source: item.source || 'Unknown',
        })
      })
    })

    // Sort by normalized date (if available)
    entities.sort((a, b) => {
      const dateA = a.normalized || ''
      const dateB = b.normalized || ''
      return dateA.localeCompare(dateB)
    })

    return entities
  }, [data])

  if (temporalData.length === 0) {
    return (
      <Box p={4} bg="gray.50" borderRadius="md">
        <Text color="gray.500">No temporal entities found</Text>
      </Box>
    )
  }

  return (
    <Box maxH="600px" overflowY="auto">
      <VStack spacing={3} align="stretch">
        {temporalData.map((entity, index) => (
          <Box
            key={index}
            p={4}
            bg="gray.50"
            borderLeft="4px"
            borderColor="blue.500"
            borderRadius="md"
          >
            <Flex justify="space-between" align="flex-start">
              <Box flex={1}>
                <HStack spacing={2} mb={1}>
                  <Text fontWeight="bold">{entity.text}</Text>
                  {entity.normalization_type && (
                    <Badge colorScheme="blue" fontSize="xs">
                      {entity.normalization_type}
                    </Badge>
                  )}
                  {entity.confidence && (
                    <Badge colorScheme="green" fontSize="xs">
                      {Math.round(entity.confidence * 100)}%
                    </Badge>
                  )}
                </HStack>
                {entity.normalized && (
                  <Text fontSize="sm" color="gray.600" mb={1}>
                    Normalized: {entity.normalized}
                  </Text>
                )}
                <Text fontSize="xs" color="gray.500">
                  Source: {entity.source} • {entity.document_title}
                </Text>
              </Box>

              {entity.reflection_scores && (
                <Box textAlign="right">
                  <Text fontSize="xs" color="gray.600" mb={1}>Quality Scores:</Text>
                  <Text fontSize="xs">R: {entity.reflection_scores.relevance.toFixed(2)}</Text>
                  <Text fontSize="xs">A: {entity.reflection_scores.accuracy.toFixed(2)}</Text>
                  <Text fontSize="xs">C: {entity.reflection_scores.consistency.toFixed(2)}</Text>
                </Box>
              )}
            </Flex>
          </Box>
        ))}
      </VStack>
    </Box>
  )
}
