/**
 * Fractal — Browser Job (Trigger.dev)
 * Background job that spins up Browserbase sessions for
 * DOM extraction, navigation, screenshots, and interactions.
 */

import { task } from "@trigger.dev/sdk/v3";
import { z } from "zod";
import {
  createSession,
  extractElements,
  extractPageData,
  closeSession,
  withRetry,
} from "../utils/browserbase";

// ── Schema Definitions ──

const DOMExtractionPayload = z.object({
  url: z.string().url(),
  selectors: z.array(z.string()).min(1).max(20),
  waitFor: z.string().nullable().optional(),
  timeoutMs: z.number().min(1000).max(120000).default(30000),
});

const NavigationPayload = z.object({
  url: z.string().url(),
  extractText: z.boolean().default(true),
  extractLinks: z.boolean().default(false),
  extractMetadata: z.boolean().default(false),
  waitFor: z.string().nullable().optional(),
});

const ScreenshotPayload = z.object({
  url: z.string().url(),
  fullPage: z.boolean().default(false),
  viewportWidth: z.number().min(320).max(3840).default(1280),
  viewportHeight: z.number().min(240).max(2160).default(720),
});

const InteractionAction = z.object({
  type: z.enum(["click", "type", "wait", "extract", "scroll"]),
  selector: z.string().optional(),
  text: z.string().optional(),
  delay: z.number().optional(),
});

const InteractionPayload = z.object({
  url: z.string().url(),
  actions: z.array(InteractionAction).min(1).max(50),
  timeoutMs: z.number().min(1000).max(120000).default(30000),
});

// ── Job: DOM Extraction ──

export const browserExtractDOM = task({
  id: "browser-extract-dom",
  retry: { maxAttempts: 2 },
  run: async (payload: z.infer<typeof DOMExtractionPayload>) => {
    const input = DOMExtractionPayload.parse(payload);
    const startTime = Date.now();

    const { browser, page } = await withRetry(() =>
      createSession({ timeoutMs: input.timeoutMs })
    );

    try {
      // Navigate to URL
      await page.goto(input.url, {
        waitUntil: "networkidle2",
        timeout: input.timeoutMs,
      });

      // Wait for specific element if requested
      if (input.waitFor) {
        await page.waitForSelector(input.waitFor, {
          timeout: input.timeoutMs,
        });
      }

      // Extract DOM elements
      const elements = await extractElements(page, input.selectors);

      return {
        url: input.url,
        elements,
        elementCount: elements.length,
        extractionTimeMs: Date.now() - startTime,
      };
    } finally {
      await closeSession(browser);
    }
  },
});

// ── Job: Page Navigation ──

export const browserNavigate = task({
  id: "browser-navigate",
  retry: { maxAttempts: 2 },
  run: async (payload: z.infer<typeof NavigationPayload>) => {
    const input = NavigationPayload.parse(payload);

    const { browser, page } = await withRetry(() => createSession());

    try {
      await page.goto(input.url, {
        waitUntil: "networkidle2",
        timeout: 30000,
      });

      if (input.waitFor) {
        await page.waitForSelector(input.waitFor, { timeout: 10000 });
      }

      const pageData = await extractPageData(page);

      return {
        title: pageData.title,
        url: pageData.url,
        textContent: input.extractText ? pageData.textContent : "",
        links: input.extractLinks ? pageData.links : [],
        metadata: input.extractMetadata ? pageData.metadata : {},
      };
    } finally {
      await closeSession(browser);
    }
  },
});

// ── Job: Screenshot Capture ──

export const browserScreenshot = task({
  id: "browser-screenshot",
  retry: { maxAttempts: 2 },
  run: async (payload: z.infer<typeof ScreenshotPayload>) => {
    const input = ScreenshotPayload.parse(payload);

    const { browser, page } = await withRetry(() =>
      createSession({
        viewportWidth: input.viewportWidth,
        viewportHeight: input.viewportHeight,
      })
    );

    try {
      await page.goto(input.url, {
        waitUntil: "networkidle2",
        timeout: 30000,
      });

      const screenshot = await page.screenshot({
        encoding: "base64",
        fullPage: input.fullPage,
        type: "png",
      });

      return {
        url: input.url,
        imageBase64: screenshot as string,
        width: input.viewportWidth,
        height: input.viewportHeight,
        fullPage: input.fullPage,
      };
    } finally {
      await closeSession(browser);
    }
  },
});

// ── Job: Multi-Step Interaction ──

export const browserInteract = task({
  id: "browser-interact",
  retry: { maxAttempts: 1 },
  run: async (payload: z.infer<typeof InteractionPayload>) => {
    const input = InteractionPayload.parse(payload);
    const results: Array<{ action: string; result: unknown }> = [];

    const { browser, page } = await withRetry(() =>
      createSession({ timeoutMs: input.timeoutMs })
    );

    try {
      await page.goto(input.url, {
        waitUntil: "networkidle2",
        timeout: input.timeoutMs,
      });

      for (const action of input.actions) {
        switch (action.type) {
          case "click":
            if (action.selector) {
              await page.click(action.selector);
              results.push({ action: `click:${action.selector}`, result: "ok" });
            }
            break;

          case "type":
            if (action.selector && action.text) {
              await page.type(action.selector, action.text, {
                delay: action.delay || 50,
              });
              results.push({ action: `type:${action.selector}`, result: "ok" });
            }
            break;

          case "wait":
            if (action.selector) {
              await page.waitForSelector(action.selector, {
                timeout: action.delay || 10000,
              });
              results.push({ action: `wait:${action.selector}`, result: "ok" });
            }
            break;

          case "extract":
            if (action.selector) {
              const elements = await extractElements(page, [action.selector]);
              results.push({
                action: `extract:${action.selector}`,
                result: elements,
              });
            }
            break;

          case "scroll":
            await page.evaluate(
              (pixels) => window.scrollBy(0, pixels),
              action.delay || 500
            );
            results.push({ action: "scroll", result: "ok" });
            break;
        }
      }

      return {
        url: input.url,
        actionsExecuted: results.length,
        results,
      };
    } finally {
      await closeSession(browser);
    }
  },
});
