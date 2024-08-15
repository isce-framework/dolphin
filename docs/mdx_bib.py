# Source: https://github.com/bobmyhill/mdx_bib/blob/6b13bbbc407617a5e93ed0f8a0e5e4c52f73f677/mdx_bib.py
import re
import string
from collections import Counter, OrderedDict
from xml.etree import ElementTree as etree

from markdown.extensions import Extension
from markdown.inlinepatterns import Pattern
from markdown.preprocessors import Preprocessor
from markdown.treeprocessors import Treeprocessor
from pybtex.database.input import bibtex
from pybtex.exceptions import PybtexError

BRACKET_RE = re.compile(r"\[([^\[]+)\]")
CITE_RE = re.compile(r"@(\w+)")
DEF_RE = re.compile(r"\A {0,3}\[@(\w+)\]:\s*(.*)")
INDENT_RE = re.compile(r"\A\t| {4}(.*)")

CITATION_RE = r"@(\w+)"


class Bibliography(object):
    """Keep track of document references and citations for exporting."""

    def __init__(self, extension, bibtex_file, order):
        self.extension = extension
        self.order = order

        self.citations = OrderedDict()
        self.references = {}

        if bibtex_file:
            try:
                parser = bibtex.Parser()
                self.bibsource = parser.parse_file(bibtex_file).entries
                self.labels = {
                    id: self.formatCitation(self.bibsource[id]) for id in self.bibsource
                }
                for value, occurrences in Counter(self.labels.values()).items():
                    if occurrences > 1:
                        for xkey, xvalue in self.labels.items():
                            i = 0
                            if xvalue == value:
                                self.labels[xkey] = (
                                    f"{xvalue}{string.ascii_lowercase[i]}"
                                )
                                i += 1

            except PybtexError:
                print("Error loading bibtex file")
                self.bibsource = {}
                self.labels = {}
        else:
            self.bibsource = {}

    def addCitation(self, citekey):
        self.citations[citekey] = self.citations.get(citekey, 0) + 1

    def setReference(self, citekey, reference):
        self.references[citekey] = reference

    def citationID(self, citekey):
        return "cite-" + citekey

    def referenceID(self, citekey):
        return "ref-" + citekey

    def formatAuthor(self, author):
        out = f"{author.last_names[0]} {author.first_names[0][0]}."
        if author.middle_names:
            out += f"{author.middle_names[0][0]}."
        return out.replace("{", "").replace("}", "")

    def formatAuthorSurname(self, author):
        out = author.last_names[0]
        return out.replace("{", "").replace("}", "")

    def formatReference(self, ref):
        author_list = list(map(self.formatAuthor, ref.persons["author"]))

        if len(author_list) == 1:
            authors = author_list[0]
        else:
            authors = ", ".join(author_list[:-1])
            authors += f" and {author_list[-1]}"

        # Harvard style
        # Surname, Initial, ... and Last_Surname,
        # Initial, Year. Title. Journal, Volume(Issue), pages. doi.

        title = ref.fields["title"].replace("{", "").replace("}", "")
        journal = ref.fields.get("journal", "")
        volume = ref.fields.get("volume", "")
        issue = ref.fields.get("issue", "")
        year = ref.fields.get("year")
        pages = ref.fields.get("pages")
        doi = ref.fields.get("doi")

        reference = f"<p>{authors}, {year}. {title}."
        if journal:
            reference += f" <i>{journal}</i>."
            if volume:
                reference += f" <i>{volume}</i>"
            if issue:
                reference += f"({issue})"
            if pages:
                reference += f", pp.{pages}"
            reference += "."
        if doi:
            reference += (
                f' <a href="https://dx.doi.org/{doi}" target="_blank">{doi}</a>'
            )
        reference += "</p>"

        return reference

    def formatCitation(self, ref):
        author_list = list(map(self.formatAuthorSurname, ref.persons["author"]))
        year = ref.fields.get("year")

        if len(author_list) == 1:
            citation = f"{author_list[0]}"
        elif len(author_list) == 2:
            citation = f"{author_list[0]} and {author_list[1]}"
        else:
            citation = f"{author_list[0]} et al."

        citation += f", {year}"

        return citation

    def makeBibliography(self, root):
        if self.order == "alphabetical":
            raise (NotImplementedError)

        div = etree.Element("div")
        div.set("class", "references")

        if not self.citations:
            return div

        table = etree.SubElement(div, "table")
        tbody = etree.SubElement(table, "tbody")
        for id in self.citations:
            tr = etree.SubElement(tbody, "tr")
            tr.set("id", self.referenceID(id))
            ref_id = etree.SubElement(tr, "td")
            ref_txt = etree.SubElement(tr, "td")
            if id in self.references:
                self.extension.parser.parseChunk(ref_txt, self.references[id])
                ref_id.text = self.labels[id]
            elif id in self.bibsource:
                ref_txt.text = self.formatReference(self.bibsource[id])
                ref_id.text = self.labels[id]
            else:
                ref_txt.text = "Missing citation"

        return div

    def clearCitations(self):
        self.citations = OrderedDict()


class CitationsPreprocessor(Preprocessor):
    """Gather reference definitions and citation keys."""

    def __init__(self, bibliography):
        self.bib = bibliography

    def subsequentIndents(self, lines, i):
        """Concatenate consecutive indented lines."""
        linesOut = []
        while i < len(lines):
            m = INDENT_RE.match(lines[i])
            if m:
                linesOut.append(m.group(1))
                i += 1
            else:
                break
        return " ".join(linesOut), i

    def run(self, lines):
        linesOut = []
        i = 0

        while i < len(lines):
            # Check to see if the line starts a reference definition
            m = DEF_RE.match(lines[i])
            if m:
                key = m.group(1)
                reference = m.group(2)
                indents, i = self.subsequentIndents(lines, i + 1)
                reference += " " + indents

                self.bib.setReference(key, reference)
                continue

            # Look for all @citekey patterns inside hard brackets
            for bracket in BRACKET_RE.findall(lines[i]):
                for c in CITE_RE.findall(bracket):
                    self.bib.addCitation(c)
            linesOut.append(lines[i])
            i += 1

        return linesOut


class CitationsPattern(Pattern):
    """Handles converting citations keys into links."""

    def __init__(self, pattern, bibliography):
        super(CitationsPattern, self).__init__(pattern)
        self.bib = bibliography

    def handleMatch(self, m):
        id = m.group(2)
        if id in self.bib.bibsource:
            a = etree.Element("a")
            a.set("id", self.bib.citationID(id))
            a.set("href", "#" + self.bib.referenceID(id))
            a.set("class", "citation")
            a.text = self.bib.labels[id]
            return a
        else:
            return None


class CitationsTreeprocessor(Treeprocessor):
    """Add a bibliography/reference section to the end of the document."""

    def __init__(self, bibliography):
        self.bib = bibliography

    def run(self, root):
        citations = self.bib.makeBibliography(root)
        root.append(citations)
        self.bib.clearCitations()


class CitationsExtension(Extension):
    def __init__(self, *args, **kwargs):
        self.config = {
            "bibtex_file": ["", "Bibtex file path"],
            "order": ["unsorted", "Order of the references (unsorted, alphabetical)"],
        }
        super(CitationsExtension, self).__init__(*args, **kwargs)
        self.bib = Bibliography(
            self,
            self.getConfig("bibtex_file"),
            self.getConfig("order"),
        )

    def extendMarkdown(self, md):
        md.registerExtension(self)
        self.parser = md.parser
        self.md = md

        md.preprocessors.register(CitationsPreprocessor(self.bib), "mdx_bib", 15)
        md.inlinePatterns.register(
            CitationsPattern(CITATION_RE, self.bib), "mdx_bib", 175
        )
        md.treeprocessors.register(CitationsTreeprocessor(self.bib), "mdx_bib", 25)


def makeExtension(*args, **kwargs):
    return CitationsExtension(*args, **kwargs)
