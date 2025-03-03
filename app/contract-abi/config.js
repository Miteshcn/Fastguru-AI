import { http, createConfig } from "@wagmi/core";
import { polygon } from "@wagmi/core/chains";

export const config = createConfig({
  chains: [polygon],
  transports: {
    // [mainnet.id]: http(),
    [polygon.id]: http(),
  },
});
