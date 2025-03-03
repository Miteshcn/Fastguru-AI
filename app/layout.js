import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap-icons/font/bootstrap-icons.css';
import { Inter } from "next/font/google";
import "./globals.css";
import Providers from "./Web3Provider"
const inter = Inter({ subsets: ["latin"] });
import "@rainbow-me/rainbowkit/styles.css";
export const metadata = {
  title: "Fastcoin",
  description: "Developed by CFCsoftware",
};
import { WalletProvider } from './context/WalletContext';
import { ActiveTabProvider } from './context/ActiveTab';
import { CrmProvider } from './context/CrmContext';
import { RealestateProvider } from "./context/RealestateContext";
export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Providers>
          <WalletProvider>
            <ActiveTabProvider>
              <CrmProvider>
                <RealestateProvider>
                  {children}
                </RealestateProvider>{" "}
              </CrmProvider>
            </ActiveTabProvider>
          </WalletProvider>
        </Providers>
        <Providers></Providers>
      </body>
    </html>
  );
}
