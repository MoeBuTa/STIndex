'use client'

import { useMemo, useCallback, useState, useEffect } from 'react'
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
} from 'reactflow'
import { Box, Text, VStack, Badge, HStack } from '@chakra-ui/react'
import { calculateCoOccurrence, SpatioTemporalEvent } from '../lib/analytics'
import 'reactflow/dist/style.css'

interface EntityNetworkProps {
  events: SpatioTemporalEvent[]
  height?: string
  minCoOccurrence?: number
}

export function EntityNetwork({
  events,
  height = '600px',
  minCoOccurrence = 2,
}: EntityNetworkProps) {
  const [isMounted, setIsMounted] = useState(false)

  // Ensure client-side only rendering
  useEffect(() => {
    setIsMounted(true)
  }, [])
  // Calculate co-occurrence network
  const { nodes: initialNodes, edges: initialEdges, categoryStats, categoryColors } = useMemo(() => {
    if (events.length === 0) {
      return { nodes: [], edges: [], categoryStats: new Map(), categoryColors: new Map() }
    }

    const coOccurrence = calculateCoOccurrence(events)

    // Create nodes from entities and track categories
    const entityCounts = new Map<string, number>()
    const entityCategories = new Map<string, Map<string, number>>() // Track all categories per entity
    const categoryStats = new Map<string, number>() // Track category usage

    events.forEach((e) => {
      const entityText = e.text
      const category = e.category || 'unknown'

      // Count entity occurrences
      entityCounts.set(entityText, (entityCounts.get(entityText) || 0) + 1)

      // Track categories for this entity
      if (!entityCategories.has(entityText)) {
        entityCategories.set(entityText, new Map())
      }
      const catMap = entityCategories.get(entityText)!
      catMap.set(category, (catMap.get(category) || 0) + 1)

      // Track overall category stats
      categoryStats.set(category, (categoryStats.get(category) || 0) + 1)
    })

    // Get top entities by frequency
    const topEntities = Array.from(entityCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20)
      .map(([text]) => text)

    // Create dynamic color mapping for all categories
    const allCategories = Array.from(categoryStats.keys()).sort()
    const colorPalette = [
      '#e53e3e', // red.500
      '#dd6b20', // orange.500
      '#3182ce', // blue.500
      '#805ad5', // purple.500
      '#38a169', // green.500
      '#d69e2e', // yellow.600
      '#319795', // teal.500
      '#d53f8c', // pink.500
      '#0bc5ea', // cyan.500
      '#ed8936', // orange.400
    ]

    const categoryColors = new Map<string, string>()
    allCategories.forEach((category, idx) => {
      if (category === 'unknown') {
        categoryColors.set(category, '#718096') // gray.600
      } else {
        categoryColors.set(category, colorPalette[idx % colorPalette.length])
      }
    })

    const nodes: Node[] = topEntities.map((entity, idx) => {
      const count = entityCounts.get(entity) || 0

      // Get most common category for this entity
      const categories = entityCategories.get(entity)!
      const mostCommonCategory = Array.from(categories.entries())
        .sort((a, b) => b[1] - a[1])[0][0]

      // Calculate node size based on frequency
      const nodeSize = Math.max(30 + count * 5, 40)

      // Get color from dynamic color map
      const color = categoryColors.get(mostCommonCategory) || '#718096'

      return {
        id: entity,
        type: 'default',
        position: {
          x: Math.cos((idx / topEntities.length) * 2 * Math.PI) * 200 + 300,
          y: Math.sin((idx / topEntities.length) * 2 * Math.PI) * 200 + 300,
        },
        data: {
          label: entity.length > 20 ? entity.substring(0, 20) + '...' : entity,
          category: mostCommonCategory,
          count: count,
        },
        style: {
          background: color,
          color: 'white',
          border: '2px solid white',
          borderRadius: '50%',
          width: nodeSize,
          height: nodeSize,
          fontSize: '10px',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          textAlign: 'center',
          padding: '4px',
        },
      }
    })

    // Create edges from co-occurrence
    const edges: Edge[] = []
    const addedEdges = new Set<string>()

    topEntities.forEach((entity1) => {
      const connections = coOccurrence.get(entity1)
      if (!connections) return

      connections.forEach((weight, entity2) => {
        if (!topEntities.includes(entity2)) return
        if (weight < minCoOccurrence) return

        const edgeId =
          entity1 < entity2 ? `${entity1}-${entity2}` : `${entity2}-${entity1}`
        if (addedEdges.has(edgeId)) return

        addedEdges.add(edgeId)
        edges.push({
          id: edgeId,
          source: entity1,
          target: entity2,
          animated: weight > 3,
          style: {
            stroke: '#94a3b8',
            strokeWidth: Math.min(weight, 5),
          },
          label: weight > 3 ? `${weight}` : '',
          labelStyle: {
            fontSize: '10px',
            fill: '#64748b',
          },
        })
      })
    })

    return { nodes, edges, categoryStats, categoryColors }
  }, [events, minCoOccurrence])

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)

  const onNodeClick = useCallback((_event: any, node: Node) => {
    console.log('Clicked node:', node)
    // Could add popup or detail view here
  }, [])

  if (!isMounted) {
    return (
      <Box p={6} bg="gray.50" borderRadius="md" height={height}>
        <VStack justify="center" h="full">
          <Text color="gray.500">Loading network...</Text>
        </VStack>
      </Box>
    )
  }

  if (events.length === 0) {
    return (
      <Box p={6} bg="gray.50" borderRadius="md" height={height}>
        <VStack justify="center" h="full">
          <Text color="gray.500">No entities available for network visualization</Text>
        </VStack>
      </Box>
    )
  }

  if (nodes.length === 0) {
    return (
      <Box p={6} bg="gray.50" borderRadius="md" height={height}>
        <VStack justify="center" h="full">
          <Text color="gray.500">
            No co-occurrence relationships found (minimum: {minCoOccurrence})
          </Text>
        </VStack>
      </Box>
    )
  }

  return (
    <Box position="relative">
      <Box height={height} border="1px solid" borderColor="gray.200" borderRadius="md">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          fitView
          attributionPosition="bottom-left"
        >
          <Background />
          <Controls />
          <MiniMap
            nodeColor={(node) => {
              const style = node.style as any
              return style?.background || '#94a3b8'
            }}
            maskColor="rgba(0, 0, 0, 0.1)"
          />
        </ReactFlow>
      </Box>

      {/* Legend - Top 10 Types */}
      <Box
        position="absolute"
        top={4}
        right={4}
        bg="white"
        p={3}
        borderRadius="md"
        shadow="md"
        zIndex={10}
        maxW="250px"
      >
        <VStack align="start" spacing={2}>
          <Text fontSize="sm" fontWeight="bold">
            Top Entity Types
          </Text>
          {Array.from(categoryStats.entries())
            .sort((a, b) => b[1] - a[1]) // Sort by frequency
            .slice(0, 10) // Show only top 10
            .map(([category, count]) => {
              const color = categoryColors.get(category) || '#718096'
              const displayName = category
                .split('_')
                .map((word: string) => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ')

              return (
                <HStack key={category} spacing={2} w="full" justify="space-between">
                  <HStack spacing={2}>
                    <Box w="12px" h="12px" borderRadius="full" bg={color} />
                    <Text fontSize="xs" noOfLines={1}>{displayName}</Text>
                  </HStack>
                  <Badge fontSize="xs" colorScheme="gray">
                    {count}
                  </Badge>
                </HStack>
              )
            })}
          {categoryStats.size > 10 && (
            <Text fontSize="xs" color="gray.500" fontStyle="italic">
              +{categoryStats.size - 10} more types
            </Text>
          )}
        </VStack>
      </Box>

      {/* Stats */}
      <Box mt={4}>
        <HStack spacing={4}>
          <Badge colorScheme="blue">{nodes.length} entities</Badge>
          <Badge colorScheme="purple">{edges.length} connections</Badge>
          <Text fontSize="xs" color="gray.600">
            Min co-occurrence: {minCoOccurrence}
          </Text>
        </HStack>
      </Box>
    </Box>
  )
}
