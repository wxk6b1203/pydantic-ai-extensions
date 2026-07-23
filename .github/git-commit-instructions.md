# Git Commit Message Instructions

## Format Specification
You must format all commit messages using the Conventional Commits specification:
`<type>(<scope>): <description>`

## Allowed Types
*   `feat`: A new feature for the application.
*   `fix`: A bug fix.
*   `docs`: Documentation changes only.
*   `style`: Changes that do not affect the meaning of the code (formatting, missing semi-colons, etc).
*   `refactor`: A code change that neither fixes a bug nor adds a feature.
*   `perf`: A code change that improves performance.
*   `test`: Adding missing tests or correcting existing tests.
*   `chore`: Changes to the build process or auxiliary tools and libraries.

## Style Constraints
*   **Length**: Keep the subject description under 200 characters.
*   **Case**: Use lowercase for the first word.
*   **Punctuation**: Do not end the subject line with a period.
*   **Mood**: Use the imperative mood (e.g., "add" instead of "added" or "adds").

## Extended Body (Optional)
*   Separate the subject line from the body with a blank line.
*   Explain the "what" and "why" of the changes, not the "how".
*   Use bullet points for lists.
