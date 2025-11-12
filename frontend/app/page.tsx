'use client'

import dynamic from 'next/dynamic'
import { Center, Spinner } from '@chakra-ui/react'

// Disable SSR for the entire dashboard to avoid Chakra UI hydration issues
const DashboardContent = dynamic(() => import('./components/DashboardContent'), {
  ssr: false,
  loading: () => (
    <Center minH="100vh" bg="gray.50">
      <Spinner size="xl" color="blue.500" thickness="4px" />
    </Center>
  ),
})

export default function Home() {
  return <DashboardContent />
}
