var app = new Vue({
    el: '#app',
    data: {
        operator: 'facet',
        dimensions: ["u", "g", "r",
            "i", "z", "petroRad_r", "s_z"],
        orderedDimensions: {},
        checkedDimension: null,
        sets: [],
        utility: null,
        inputSet: null,
        webService: '',
        selectedSetId: null,
        loading: false,
        history: [],
        loadSteps: [],
        getscores: false,
        predictedScores: [],
        getPredictedScores: false,
        seenPredicates: [],
        novelty: null,
        curiosityRatio: 0,
        trainingSet: "Scattered",
        guidanceMode: "Partially guided"
    },
    mounted() {
        this.loading = true
        this.webService = window.location.origin + "/operators/"
        const url = new URL(this.webService + "get-dataset-information")
        axios.get(url).then(response => {
            this.dimensions = response.data.dimensions.map((x) => x.replace('galaxies.', ''))
            this.orderedDimensions = {}
            _.forIn(response.data.ordered_dimensions, (value, key) => {
                this.orderedDimensions[key.replace('galaxies.', '')] = value
            })
            this.loading = false
        })
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl)
        })
    },
    computed: {
        facetDimensions: function () {
            if (this.selectedSetId == null) {
                return this.dimensions
            } else {
                return _.difference(this.dimensions, this.sets.find((x) => x.id === this.selectedSetId).predicate.map((x) => x.dimension.replace('galaxies.', '')))
            }
        },
        neighborsDimensions: function () {
            if (this.selectedSetId == null) {
                return Object.keys(this.orderedDimensions)
            } else {
                return _.intersection(Object.keys(this.orderedDimensions), this.sets.find((x) => x.id === this.selectedSetId).predicate.map((x) => x.dimension.replace('galaxies.', '')))
            }
        },
        sortedSets: function () {
            if (this.history.length > 0) {
                lastAction = this.history[this.history.length - 1]
                if (lastAction.checkedDimension) {
                    var sortDimension = lastAction.checkedDimension
                    var sortDimensionValues = this.orderedDimensions[sortDimension]
                    if (sortDimensionValues && sortDimensionValues.length > 0) {
                        setComparison = function (set1, set2) {
                            return sortDimensionValues.indexOf(set1.predicate.find(x => x.dimension == sortDimension).value) - sortDimensionValues.indexOf(set2.predicate.find(x => x.dimension == sortDimension).value)
                        }
                        return this.sets.sort(setComparison)
                    } else {
                        return this.sets
                    }
                }
            }
            return this.sets
        },
        saveLink: function () {
            return 'data:attachment/json,' + encodeURI(JSON.stringify(this.history))
        },
        selectedSetPredicateCount: function () {
            if (this.selectedSetId) {
                return this.sets.find((x) => x.id === this.selectedSetId).predicate.length
            }
            return 0
        },
        isLoading: function () {
            if (this.loadSteps.length == 0) {
                return false
            } else if (this.inputSet != this.loadSteps[0].inputSet |
                this.selectedSetId != this.loadSteps[0].selectedSetId |
                this.operator != this.loadSteps[0].operator |
                this.checkedDimension != this.loadSteps[0].checkedDimension) {
                this.loadSteps = []
                this.history = []
                return false
            } else {
                return true
            }
        }
    },
    methods: {
        url: function (item) {
            return `http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=${item.ra}&dec=${item.dec}&scale=0.4&width=120&height=120`
        },
        set_id: function (set) {
            return set.id < 0 ? "No Id " + Math.abs(set.id) : set.id
        },
        submitted() {
            if (this.sets.length !== 0 & this.selectedSetId === null) {
                alert("Please click anywhere on a set to select the next input set.")
            } else if (this.operator === 'facet' & !this.checkedDimension) {
                alert("Please select at least one dimension.")
            } else {
                this.loading = true
                this.inputSet = _.find(this.sets, (set) => set.id === this.selectedSetId)
                const url = new URL(this.webService + "by-" + this.operator + '-g')
                url.searchParams.append("get_scores", this.getscores)
                url.searchParams.append("get_predicted_scores", this.getPredictedScores)
                this.seenPredicates.forEach(element => {
                    url.searchParams.append("seen_predicates", element)
                });
                if (!this.inputSet) {
                    url.searchParams.append("input_set_id", -1)
                    this.inputSet = null
                } else {
                    url.searchParams.append("input_set_id", this.inputSet.id)
                }
                if (this.operator === 'facet' | this.operator === 'neighbors') {
                    url.searchParams.append("dimensions", ["galaxies." + this.checkedDimension])
                }

                axios.get(url).then(response => {
                    if (this.inputSet)
                        delete this.inputSet.data
                    this.history.push({
                        "selectedSetId": this.selectedSetId,
                        "operator": this.operator,
                        "checkedDimension": this.checkedDimension,
                        "url": url,
                        "inputSet": this.inputSet
                    })
                    this.checkedDimension = null
                    this.selectedSetId = null
                    response.data.sets.forEach(set => set.predicate.forEach(item => item.dimension = item.dimension.replace('galaxies.', '')))
                    this.sets = response.data.sets
                    this.utility = response.data.utility
                    this.novelty = response.data.novelty
                    this.seenPredicates = response.data.seenPredicates
                    this.predictedScores = response.data.predictedScores
                    this.loadStep(false)
                    this.loading = false
                    historyCardIndex = this.history.length-1
                    setTimeout(function() {document.getElementById("operation-"+historyCardIndex).scrollIntoView()}, 0)
                })
            }
        },
        undo() {
            this.loading = true
            this.operator = "undo"
            this.history.pop()
            previous_state = this.history[this.history.length - 1]
            axios.get(previous_state.url).then(response => {
                this.inputSet = previous_state.inputSet
                this.checkedDimension = null
                this.selectedSetId = null
                response.data.sets.forEach(set => set.predicate.forEach(item => item.dimension = item.dimension.replace('galaxies.', '')))
                this.sets = response.data.sets
                this.utility = response.data.utility
                this.novelty = response.data.novelty
                this.seenPredicates = response.data.seenPredicates
                this.predictedScores = response.data.predictedScores
                this.operator = previous_state.operator
                this.loading = false
            })
        },
        restart() {
            this.operator = 'facet'
            this.inputSet = null
            this.selectedSetId = null
            this.sets = []
            this.history = []
            this.utility = null
            this.predictedScores = []
            this.novelty = null
            this.seenPredicates = []
        },
        loadPipeline(event) {
            if (event.target.files.length != 0) {
                const reader = new FileReader();
                reader.addEventListener('load', (readEvent) => {
                    console.log(readEvent.target.result)
                    this.loadSteps = JSON.parse(readEvent.target.result)
                    this.sets = []
                    this.loadStep(true)
                    event.target.value = ""
                })
                reader.readAsText(event.target.files[0])
            }
        },
        loadStep(isFirstStep) {
            if (this.loadSteps.length > 0) {
                if (!isFirstStep) {
                    this.loadSteps.shift()
                }
                if (this.loadSteps.length > 0) {
                    this.inputSet = this.loadSteps[0].inputSet
                    this.selectedSetId = this.loadSteps[0].selectedSetId
                    this.operator = this.loadSteps[0].operator
                    this.checkedDimension = this.loadSteps[0].checkedDimension
                }
            }
        },
        setClass(id) {
            if (this.setClicked === id) {
                return "row selected"
            }
            else {
                return "row"
            }
        },
        setClicked(set) {
            if (set.id >= 0)
                this.selectedSetId = set.id
        },
        loadPrediction(setId, operation, attribute) {
            this.selectedSetId = setId
            this.operator = operation
            this.checkedDimension = attribute
        },
        isSelectedPrediction(setId, operation, attribute) {
            if (operation == "facet" || operation == "neighbors") {
                return this.selectedSetId == setId && this.operator == operation && this.checkedDimension == attribute
            } else {
                return this.selectedSetId == setId && this.operator == operation
            }
        }
    }
})