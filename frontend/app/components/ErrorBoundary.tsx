'use client'

import { Component, ReactNode } from 'react'
import { Box, Text, Button, VStack } from '@chakra-ui/react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error?: Error
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('ErrorBoundary caught an error:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <Box p={6} bg="red.50" borderRadius="md" border="1px solid" borderColor="red.200">
          <VStack spacing={4} align="start">
            <Text fontWeight="bold" color="red.700">
              Something went wrong
            </Text>
            <Text fontSize="sm" color="red.600">
              {this.state.error?.message || 'An error occurred while rendering this component'}
            </Text>
            <Button
              size="sm"
              colorScheme="red"
              onClick={() => this.setState({ hasError: false, error: undefined })}
            >
              Try again
            </Button>
          </VStack>
        </Box>
      )
    }

    return this.props.children
  }
}
