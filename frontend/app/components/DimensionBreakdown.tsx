'use client'

import { useMemo, useState } from 'react'
import { Box, Text, VStack, HStack, Badge, Flex, Button, Progress } from '@chakra-ui/react'

interface DimensionEntity {
  text: string
  category?: string
  confidence?: number
  dimension_name?: string
  [key: string]: any
}

interface ExtractionResult {
  chunk_id: string
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
    document_metadata: {
      source: string
      category: string
    }
  }
}

interface DimensionBreakdownProps {
  data: ExtractionResult[]
}

export function DimensionBreakdown({ data }: DimensionBreakdownProps) {
  const [activeTab, setActiveTab] = useState<string>('')

  // Extract all custom dimensions (not temporal/spatial)
  const dimensions = useMemo(() => {
    const dimensionMap = new Map<string, DimensionEntity[]>()

    data.forEach((item) => {
      // First, check extraction.entities object
      if (item.extraction.entities) {
        Object.entries(item.extraction.entities).forEach(([dimName, entities]) => {
          // Skip temporal and spatial as they're handled separately
          if (dimName === 'temporal' || dimName === 'spatial' || !entities) {
            return
          }
          if (!dimensionMap.has(dimName)) {
            dimensionMap.set(dimName, [])
          }
          entities.forEach((entity) => {
            dimensionMap.get(dimName)?.push({
              ...entity,
              source: item.extraction.document_metadata.source,
              document_title: item.document_title,
            } as DimensionEntity)
          })
        })
      }

      // Also check for top-level dimension arrays (event_type, disease, venue_type, etc.)
      const customDimensions = ['event_type', 'disease', 'venue_type']
      customDimensions.forEach((dimName) => {
        const entities = (item.extraction as any)[dimName]
        if (entities && Array.isArray(entities) && entities.length > 0) {
          if (!dimensionMap.has(dimName)) {
            dimensionMap.set(dimName, [])
          }
          entities.forEach((entity) => {
            // Check if this entity already exists (to avoid duplicates from entities object)
            const existing = dimensionMap.get(dimName)?.find(
              (e) => e.text === entity.text && (e as any).source === item.extraction.document_metadata.source
            )
            if (!existing) {
              dimensionMap.get(dimName)?.push({
                ...entity,
                source: item.extraction.document_metadata.source,
                document_title: item.document_title,
              } as DimensionEntity)
            }
          })
        }
      })
    })

    return dimensionMap
  }, [data])

  // Set initial active tab
  useMemo(() => {
    if (dimensions.size > 0 && !activeTab) {
      setActiveTab(Array.from(dimensions.keys())[0])
    }
  }, [dimensions, activeTab])

  // Calculate category frequencies for a dimension
  const getCategoryStats = (entities: DimensionEntity[]) => {
    const categoryCount = new Map<string, number>()
    entities.forEach((entity) => {
      const category = entity.category || entity.text || 'unknown'
      categoryCount.set(category, (categoryCount.get(category) || 0) + 1)
    })

    const total = entities.length
    return Array.from(categoryCount.entries())
      .map(([category, count]) => ({
        category,
        count,
        percentage: (count / total) * 100,
      }))
      .sort((a, b) => b.count - a.count)
  }

  if (dimensions.size === 0) {
    return (
      <Box p={4} bg="gray.50" borderRadius="md">
        <Text color="gray.500">No custom dimensions found</Text>
      </Box>
    )
  }

  const formatDimensionName = (name: string) => {
    return name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
  }

  return (
    <Box>
      {/* Tab navigation */}
      <Flex flexWrap="wrap" gap={2} mb={6} borderBottom="1px" borderColor="gray.200">
        {Array.from(dimensions.keys()).map((dimName) => (
          <Button
            key={dimName}
            onClick={() => setActiveTab(dimName)}
            px={4}
            py={2}
            fontWeight="medium"
            fontSize="sm"
            variant="ghost"
            borderBottom="2px"
            borderColor={activeTab === dimName ? 'blue.500' : 'transparent'}
            color={activeTab === dimName ? 'blue.600' : 'gray.600'}
            _hover={{ color: 'gray.900' }}
            borderRadius={0}
          >
            {formatDimensionName(dimName)}
            <Badge ml={2} colorScheme="gray">
              {dimensions.get(dimName)?.length || 0}
            </Badge>
          </Button>
        ))}
      </Flex>

      {/* Tab content */}
      {Array.from(dimensions.entries()).map(([dimName, entities]) => {
        if (dimName !== activeTab) return null

        const categoryStats = getCategoryStats(entities)

        return (
          <VStack key={dimName} spacing={6} align="stretch">
            {/* Category distribution */}
            <Box>
              <Text fontSize="lg" fontWeight="bold" mb={3}>Category Distribution</Text>
              <VStack spacing={2} align="stretch">
                {categoryStats.map((stat) => (
                  <Box key={stat.category}>
                    <Flex justify="space-between" mb={1}>
                      <Text fontSize="sm" fontWeight="medium">{stat.category}</Text>
                      <HStack spacing={2}>
                        <Text fontSize="sm" color="gray.600">{stat.count}</Text>
                        <Text fontSize="sm" color="gray.500">
                          ({stat.percentage.toFixed(1)}%)
                        </Text>
                      </HStack>
                    </Flex>
                    <Box w="full" bg="gray.200" borderRadius="md" overflow="hidden" h={2}>
                      <Box
                        h="full"
                        bg="blue.500"
                        transition="all 0.3s"
                        w={`${stat.percentage}%`}
                      />
                    </Box>
                  </Box>
                ))}
              </VStack>
            </Box>

            {/* Entity list */}
            <Box>
              <Text fontSize="lg" fontWeight="bold" mb={3}>All Entities</Text>
              <VStack spacing={2} align="stretch">
                {entities.map((entity, index) => (
                  <Box
                    key={index}
                    p={3}
                    bg="gray.50"
                    borderRadius="md"
                    borderLeft="3px"
                    borderColor="blue.400"
                  >
                    <Flex justify="space-between" align="flex-start">
                      <Box>
                        <HStack spacing={2} mb={1}>
                          <Text fontWeight="medium">{entity.text}</Text>
                          {entity.category && (
                            <Badge colorScheme="purple" fontSize="xs">
                              {entity.category}
                            </Badge>
                          )}
                          {entity.confidence && (
                            <Badge colorScheme="green" fontSize="xs">
                              {Math.round(entity.confidence * 100)}%
                            </Badge>
                          )}
                        </HStack>
                        <Text fontSize="xs" color="gray.500">
                          {(entity as any).source} • {(entity as any).document_title}
                        </Text>
                      </Box>
                    </Flex>
                  </Box>
                ))}
              </VStack>
            </Box>
          </VStack>
        )
      })}
    </Box>
  )
}
