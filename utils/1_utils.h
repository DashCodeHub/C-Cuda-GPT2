//purpose of utils.h file (High Level)
/*
 - File operations
 - Memory allocation
 - Socket closing 
 - Directory operations
 - Simple string checks
 - Token value checking (for LLM token vocab)
 - Error checking (prints the bug)
*/ 

/* IF YOU DONOT KNOW C (Optional)
ifndef ('if not defined') is a preprocessor directive that says ,If a macro or symbol has NOT been defined yet, 
then compile the following block of code.prevents double inclusion of the same header file
*/

#ifndef UTILS_H 

/* IF YOU DONOT KNOW C (Optional)
A macro is basically a text replacement rule, handles by C preprocessor before the actual compilation starts
In C, a macro is defined by using #define
e.g. #define NAME replacement_text
When the compiler sees NAME, it replaces it with replacement_text before compiling.
It's like copy-paste at compile time — not a function call, not a variable — pure text replacement.
*/
#define UTILS_H

#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

// Since the code I am running is in linux machine, I need dirent.h for `opendir` `readdir` etc.
#ifndef _WIN32
#include <dirent.h>
#include <arpa/inet.h>
#endif

//function for safer version of standard c function `fopen()` to open a file
//It tries to open a file, if it fails, it prints and error message and exits the program
//If scuccess, returns the file like a normal pointer
/* IF YOU DONOT KNOW C (Optional)
extern -> Tells the compiler: "this function might be used across different .c / .cu files (translation units)"
inline -> Suggests to the compiler: "instead of making a real function call, paste the function code during compilation" (for performance)
FILE * -> FILE is a standard C structure that represents an open file (from stdio.h), and * means you're returning a pointer to FILE
*/
extern inline FILE *fopen_check(const char *path, const char *mode, const char *file, int line) {
    FILE *fp = fopen(path, mode);
    if (fp == NULL) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    return fp;
}

//Now we are defineing a macro called fopencheck and replacing the above rewritting code directly here
#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)


// function for a safer version of standard c function `fread()`  to read a file [Similar to above `fopen_check`]
extern inline void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line) {
    size_t result = fread(ptr, size, nmemb, stream);
    if (result != nmemb) {
        if (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, line);
        } else if (ferror(stream)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, line);
        } else {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, line, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

// and again define the macro for the above function [Will continue the same pattern for rest of the utility function]
#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)


//safer function to close the file
extern inline void fclose_check(FILE *fp, const char *file, int line) {
    if (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

/* IF YOU DONOT KNOW C (Optional)
In C, a socket is just an integer number (int sockfd) —
like a file descriptor — created by calling socket().
*/
// sclose_check() - takes a socket file descripter [integer] and tries to close the socket safely
extern inline void sclose_check(int sockfd, const char *file, int line) {
    if (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

// fseek_check safely moves the file pointer inside a file, and if it fails, it prints a detailed error 
// message and exits the program immediately.
extern inline void fseek_check(FILE *fp, long off, int whence, const char *file, int line) {
    if (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

// functio to write a block o fdata into an open file pointer passed to it
extern inline void fwrite_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line) {
    size_t result = fwrite(ptr, size, nmemb, stream); // fwrite() writes a block of memory into an open file; 
    // nmemb-number of items to write; stream- open file pointer where you want to write
    if (result != nmemb) {
        if (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, line);
        } else if (ferror(stream)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, line);
        } else {
            fprintf(stderr, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, line, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Written elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define fwriteCheck(ptr, size, nmemb, stream) fwrite_check(ptr, size, nmemb, stream, __FILE__, __LINE__)


// function allocate memory (safer version with printing the error)
extern inline void *malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

// to check if all the tokens are  within the range '>0' and '< vocabulary size'
extern inline void token_check(const int* tokens, int token_count, int vocab_size, const char *file, int line) {
    for(int i = 0; i < token_count; i++) {
        if(!(0 <= tokens[i] && tokens[i] < vocab_size)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, line);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", line);
            fprintf(stderr, "  Token: %d\n", tokens[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_size);
            exit(EXIT_FAILURE);
        }
    }
}
#define tokenCheck(tokens, count, vocab) token_check(tokens, count, vocab, __FILE__, __LINE__)

// to create directory of the directory doesnot exists
extern inline void create_dir_if_not_exists(const char *dir) {
    if (dir == NULL) { return; }
    struct stat st = {0};
    if (stat(dir, &st) == -1) {
        if (mkdir(dir, 0700) == -1) {
            printf("ERROR: could not create directory: %s\n", dir);
            exit(EXIT_FAILURE);
        }
        printf("created directory: %s\n", dir);
    }
}

// to scan a log directory and find all the files whose names start with `DONE_`
//extract the step numbers from their filenames, and return the highest step found
// it will be useful to check the last step completed and then resume the training

extern inline int find_max_step(const char* output_log_dir) {
    if (output_log_dir == NULL) { return -1; }
    DIR* dir;
    struct dirent* entry; // decalre a ponter 'entry' to point 'struct_dirent'
    /* IF YOU DONOT KNOW C (Optional)
    struct dirent is a standard C structure defined in <dirent.h>.
    It represents a single file or directory entry inside a folder.
    When you open a directory (opendir) and read its contents (readdir),
    each time you call readdir, it returns a pointer to a struct dirent describing one file or folder.
    */
    int max_step = -1;
    dir = opendir(output_log_dir);
    if (dir == NULL) { return -1; }
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "DONE_", 5) == 0) { //`strncmp`: compare a string [n characters]
            int step = atoi(entry->d_name + 5);
            if (step > max_step) {
                max_step = step;
            }
        }
    }
    closedir(dir);
    return max_step;
}

// to check if the string ends with .bin
extern inline int ends_with_bin(const char* str) {
    if (str == NULL) { return 0; }
    size_t len = strlen(str);
    const char* suffix = ".bin";
    size_t suffix_len = strlen(suffix);
    if (len < suffix_len) { return 0; }
    int suffix_matches = strncmp(str + len - suffix_len, suffix, suffix_len) == 0;
    return suffix_matches;
}

// The above three we are not defining them as macros these are complex function (not a simple text substitutions)

#endif