import {
    ChakraProvider,
    Container,
    VStack,
    Heading,
    Text,
    Textarea,
    Button,
    Alert,
    Box,
    List,
    ListItem,
    Progress,
    useToast,
} from "@chakra-ui/react";
import { WarningTwoIcon, CheckCircleIcon } from "@chakra-ui/icons"; // Changed icons
import { useState } from "react";
import axios from "axios";

function App() {
    const [email, setEmail] = useState(
        "Hi John, let's meet tomorrow at 10 AM to discuss the project. Please bring the report. Thanks!"
    );
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const toast = useToast();

    const handleCheck = async () => {
        if (!email.trim()) {
            toast({
                title: "Error",
                description: "Please enter an email text to classify.",
                status: "warning",
                duration: 3000,
                isClosable: true,
            });
            return;
        }

        setLoading(true);
        try {
            const response = await axios.post("http://localhost:8000/predict", {
                email_text: email,
            });
            setResult(response.data);
        } catch (error) {
            toast({
                title: "Error",
                description:
                    error.response?.data?.detail || "Failed to process email",
                status: "error",
                duration: 3000,
                isClosable: true,
            });
        } finally {
            setLoading(false);
        }
    };

    return (
        <ChakraProvider>
            <Container maxW="container.md" py={8}>
                <VStack spacing={6} align="stretch">
                    <Heading>Spam Email Detector</Heading>
                    <Text>
                        Enter an email text to check if it's spam or ham.
                    </Text>

                    <Textarea
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="Type your email here..."
                        size="lg"
                        minH="200px"
                    />

                    <Button
                        colorScheme="blue"
                        onClick={handleCheck}
                        isLoading={loading}
                        loadingText="Checking..."
                    >
                        Check
                    </Button>

                    {result && (
                        <VStack spacing={4} align="stretch">
                            <Alert
                                status={
                                    result.label === "Spam"
                                        ? "error"
                                        : "success"
                                }
                                variant="subtle"
                                borderRadius="md"
                            >
                                {result.label === "Spam" ? (
                                    <WarningTwoIcon color="red.500" />
                                ) : (
                                    <CheckCircleIcon color="green.500" />
                                )}
                                <Text fontWeight="bold" ml={2}>
                                    Prediction: {result.label}
                                </Text>
                            </Alert>

                            <Box>
                                <Text mb={2}>
                                    Confidence:{" "}
                                    {(result.confidence * 100).toFixed(2)}%
                                </Text>
                                <Progress
                                    value={result.confidence * 100}
                                    colorScheme={
                                        result.label === "Spam"
                                            ? "red"
                                            : "green"
                                    }
                                />
                            </Box>

                            <Box>
                                <Text fontWeight="bold" mb={2}>
                                    Top contributing words:
                                </Text>
                                <List spacing={2}>
                                    {result.top_words.map(([word, freq]) => (
                                        <ListItem key={word}>
                                            • {word}: {freq}
                                        </ListItem>
                                    ))}
                                </List>
                            </Box>
                        </VStack>
                    )}
                </VStack>
            </Container>
        </ChakraProvider>
    );
}

export default App;
