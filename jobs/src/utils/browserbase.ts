/**
 * Fractal — Browserbase SDK Helpers
 * Session management, error handling, and retry logic
 * for headless browser automation.
 */

import Browserbase from "@browserbasehq/sdk";
import puppeteer, { Browser, Page } from "puppeteer-core";

const BROWSERBASE_API_KEY = process.env.BROWSERBASE_API_KEY || "";
const BROWSERBASE_PROJECT_ID = process.env.BROWSERBASE_PROJECT_ID || "";

/** Configuration for a browser session */
interface SessionConfig {
  viewportWidth?: number;
  viewportHeight?: number;
  timeoutMs?: number;
  userAgent?: string;
}

/** Result from DOM element extraction */
interface DOMElement {
  selector: string;
  tagName: string;
  textContent: string;
  attributes: Record<string, string>;
  innerHTML: string;
  boundingBox: { x: number; y: number; width: number; height: number } | null;
}

/** Result from page navigation */
interface PageData {
  title: string;
  url: string;
  textContent: string;
  links: Array<{ href: string; text: string }>;
  metadata: Record<string, string>;
}

/**
 * Create a Browserbase session and connect via Puppeteer.
 */
async function createSession(
  config: SessionConfig = {}
): Promise<{ browser: Browser; page: Page; sessionId: string }> {
  const bb = new Browserbase({
    apiKey: BROWSERBASE_API_KEY,
  });

  const session = await bb.sessions.create({
    projectId: BROWSERBASE_PROJECT_ID,
  });

  const browser = await puppeteer.connect({
    browserWSEndpoint: `wss://connect.browserbase.com?apiKey=${BROWSERBASE_API_KEY}&sessionId=${session.id}`,
  });

  const pages = await browser.pages();
  const page = pages[0] || (await browser.newPage());

  // Set viewport
  await page.setViewport({
    width: config.viewportWidth || 1280,
    height: config.viewportHeight || 720,
  });

  // Set default timeout
  page.setDefaultTimeout(config.timeoutMs || 30000);

  return { browser, page, sessionId: session.id };
}

/**
 * Extract DOM elements by CSS selectors.
 */
async function extractElements(
  page: Page,
  selectors: string[]
): Promise<DOMElement[]> {
  const elements: DOMElement[] = [];

  for (const selector of selectors) {
    const matches = await page.$$(selector);

    for (const handle of matches) {
      const element = await page.evaluate((el) => {
        const attrs: Record<string, string> = {};
        for (const attr of el.attributes) {
          attrs[attr.name] = attr.value;
        }

        const rect = el.getBoundingClientRect();

        return {
          tagName: el.tagName.toLowerCase(),
          textContent: (el.textContent || "").trim().substring(0, 5000),
          attributes: attrs,
          innerHTML: el.innerHTML.substring(0, 10000),
          boundingBox:
            rect.width > 0
              ? {
                  x: Math.round(rect.x),
                  y: Math.round(rect.y),
                  width: Math.round(rect.width),
                  height: Math.round(rect.height),
                }
              : null,
        };
      }, handle);

      elements.push({
        selector,
        ...element,
      });

      await handle.dispose();
    }
  }

  return elements;
}

/**
 * Extract full page data including text, links, and metadata.
 */
async function extractPageData(page: Page): Promise<PageData> {
  return page.evaluate(() => {
    // Extract visible text
    const textContent = document.body?.innerText?.substring(0, 50000) || "";

    // Extract links
    const anchors = Array.from(document.querySelectorAll("a[href]"));
    const links = anchors
      .map((a) => ({
        href: (a as HTMLAnchorElement).href,
        text: (a.textContent || "").trim().substring(0, 200),
      }))
      .filter((l) => l.href && l.text)
      .slice(0, 100);

    // Extract meta tags
    const metaTags = Array.from(document.querySelectorAll("meta"));
    const metadata: Record<string, string> = {};
    for (const meta of metaTags) {
      const name = meta.getAttribute("name") || meta.getAttribute("property");
      const content = meta.getAttribute("content");
      if (name && content) {
        metadata[name] = content.substring(0, 500);
      }
    }

    return {
      title: document.title,
      url: window.location.href,
      textContent,
      links,
      metadata,
    };
  });
}

/**
 * Safely close a browser session with retry logic.
 */
async function closeSession(browser: Browser): Promise<void> {
  try {
    await browser.close();
  } catch (error) {
    console.warn("[Browserbase] Error closing session:", error);
    // Force disconnect
    browser.disconnect();
  }
}

/**
 * Retry wrapper with exponential backoff.
 */
async function withRetry<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  baseDelayMs: number = 1000
): Promise<T> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      if (attempt < maxRetries) {
        const delay = baseDelayMs * Math.pow(2, attempt);
        console.warn(
          `[Browserbase] Attempt ${attempt + 1} failed, retrying in ${delay}ms...`
        );
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }

  throw lastError;
}

export {
  createSession,
  extractElements,
  extractPageData,
  closeSession,
  withRetry,
  DOMElement,
  PageData,
  SessionConfig,
};
