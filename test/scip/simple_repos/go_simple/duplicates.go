package main

// Process is a package-level function.
func Process(data string) string {
	return data
}

// Processor has a Process method (same name as package func).
type Processor struct {
	Name string
}

// Process is a method on Processor.
func (p *Processor) Process(data string) string {
	return p.Name + ": " + data
}

// AnotherProcessor also has Process method.
type AnotherProcessor struct {
	Prefix string
}

// Process is a method on AnotherProcessor.
func (a *AnotherProcessor) Process(data string) string {
	return a.Prefix + data
}
