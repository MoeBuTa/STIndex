"use client";

import { ChakraProvider } from "@chakra-ui/react";
import { useEffect, useState } from "react";

export function Providers({ children }: { children: React.ReactNode }) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Prevent hydration mismatch by only rendering after mount
  if (!mounted) {
    return null;
  }

  return (
    <ChakraProvider resetCSS disableGlobalStyle={false}>
      {children}
    </ChakraProvider>
  );
}
